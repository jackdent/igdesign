import os
import copy
import re
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, EsmForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from igdesign.tokenization import AA_TO_IDX, IDX_TO_AA
from igdesign.model import Model
from igdesign.models.reformer import CrossAttentionRoformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def shuffle_tensor(input_tensor):
    """Randomly shuffles a tensor along its first dimension"""
    shuffled_idxs = torch.randperm(input_tensor.shape[0])
    return input_tensor[shuffled_idxs]


def get_decode_order(cfg, batch):
    """
    Get the order in which the model decodes.
    First, grab the design regions and shuffle each design region. Concatenate them according to the region_design_order.
    Second, shuffle the non-design regions.
    The decode order is the shuffled non-design regions followed by the shuffled design regions.
    Order is unique along the batch dimension, i.e. if you use batch size 10 you will get 10 different shuffle orders.
    """
    # Get the design positions (list of lists)
    design_decoding_positions = [
        cfg["regions"][region]["offset_positions"] for region in cfg["region_order"]
    ]
    all_design_positions = torch.cat(design_decoding_positions)

    # Get the non-design positions
    batch_size, total_num_positions = batch["tokenized_sequences"].shape
    non_design_positions = torch.LongTensor(
        [idx for idx in range(total_num_positions) if idx not in all_design_positions]
    )

    # Make batch_size decoding orders. Shuffle the non-design positions, then each of the design regions in order.
    # Finally, concat these to create the decoding order.
    decode_order = list()
    for batch_idx in range(batch_size):
        shuffled_non_design_positions = shuffle_tensor(non_design_positions)
        shuffled_design_positions = torch.cat(
            [
                shuffle_tensor(region_positions)
                for region_positions in design_decoding_positions
            ]
        )
        decode_order.append(
            torch.cat((shuffled_non_design_positions, shuffled_design_positions))
        )

    num_design_positions = all_design_positions.shape[0]
    offset = non_design_positions.shape[0]
    decode_order = torch.stack(decode_order)
    return decode_order, offset, num_design_positions


def batch_decode(token_ids):
    """Decode a batch of token ids (batch_size x seq_len) to a list of strings."""
    sequences = list()
    for batch_idx in range(len(token_ids)):
        sequences.append(
            "".join([IDX_TO_AA[t] for t in token_ids[batch_idx].cpu().numpy().tolist()])
        )
    return sequences


def decode_logits_greedy(logits):
    token_ids = torch.argmax(logits, dim=-1)
    if token_ids.ndim == 1:
        token_ids = token_ids[None, :]
    return batch_decode(token_ids)


def load_tokenizer(cfg):
    return AutoTokenizer.from_pretrained(
        cfg["pretrained_language_model_name_or_path"], padding="max_length"
    )


def load_language_model(cfg):
    model = EsmForMaskedLM.from_pretrained(
        cfg["pretrained_language_model_name_or_path"]
    )
    return model


def maybe_add_bos_eos_ids(input_ids, tokenizer):
    # This is done to handle the fact that GPT2Tokenizer does not add bos/eos tokens, but ProGen is trained with them.
    if tokenizer.__class__.__name__ in ["ProGenTokenizer", "GPT2TokenizerFast"]:
        bos_ids = torch.full(
            size=(input_ids.shape[0], 1),
            fill_value=tokenizer.bos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        eos_ids = torch.full(
            size=(input_ids.shape[0], 1),
            fill_value=tokenizer.eos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.concat([bos_ids, input_ids, eos_ids], dim=1)
    else:
        return input_ids


# Automatically sample last dimension from 3-dimensions
def sampler(logits, num_samples):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    samples = batch_decode(
        torch.LongTensor(
            list(torch.utils.data.WeightedRandomSampler(probs, num_samples))
        )
    )
    return samples


def freeze_network(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


class Bottleneck(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super(Bottleneck, self).__init__()
        hidden_dim = embed_dim // 2 if hidden_dim is None else embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, input_x):
        x = self.fc1(input_x)
        x = self.gelu(x)
        x = self.fc2(x)
        x += input_x
        return x


class StructuralAdapter(nn.Module):
    def __init__(
        self,
        cfg,
        lm_embed_dim,
        node_embed_dim,
    ):
        super(StructuralAdapter, self).__init__()
        self.cfg = cfg
        if cfg["structure_enc_query"]:
            self.query_project = torch.nn.Linear(node_embed_dim, lm_embed_dim)
        elif cfg["language_enc_query"]:
            self.key_project = torch.nn.Linear(node_embed_dim, lm_embed_dim)
            self.value_project = torch.nn.Linear(node_embed_dim, lm_embed_dim)
        if self.cfg["use_rope"]:
            self.attention = CrossAttentionRoformer(
                lm_embed_dim, heads=cfg["num_attention_heads"]
            )
        else:
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=lm_embed_dim,
                num_heads=cfg["num_attention_heads"],
                vdim=lm_embed_dim,
                kdim=lm_embed_dim,
            )

        self.bottleneck = Bottleneck(embed_dim=lm_embed_dim)

    # query: batch x seq_len x node_dim
    # key, value: batch x seq_len x lm_embed_dim
    def forward(self, query, key, value):
        if self.cfg["structure_enc_query"]:
            query = self.query_project(query)
        elif self.cfg["language_enc_query"]:
            key = self.key_project(key)
            value = self.value_project(value)

        if self.cfg["use_rope"]:
            x = self.attention.forward(query=query, key_value=key)
        else:
            x = self.attention.forward(query=query, key=key, value=value)[0]
        x = self.bottleneck(x)
        return x


class LMDesignModel(Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
        self.check_configs(cfg)

        self.structure_model = kwargs["structure_model"]

        structural_embed_dim = self.structure_model.res_dim_hidden
        self.language_model = load_language_model(cfg)
        self.tokenizer = load_tokenizer(cfg)
        self.structural_adapter = StructuralAdapter(
            cfg=cfg,
            lm_embed_dim=self.language_model.config.hidden_size,
            node_embed_dim=structural_embed_dim,
        )
        self.structural_adapter.train()

        if self.cfg["n_CMLM_recycles"]:
            self.original_lm_head = copy.deepcopy(self.language_model.lm_head)
            self.original_lm_head.eval()

        if cfg["freeze_structure_model"]:
            freeze_network(self.structure_model)
        if cfg["freeze_language_model"]:
            freeze_network(self.language_model)

        if (self.cfg["recycle_logits"] and self.cfg["n_recycles"] > 0) or self.cfg[
            "learn_linear_comb_embeddings"
        ]:
            vocab_size = self.language_model.config.vocab_size
            self.probs_project = nn.Sequential(
                nn.Linear(vocab_size, vocab_size // 2),
                nn.ReLU(),
                nn.Linear(vocab_size // 2, vocab_size),
            )

        self.model_type = "esm"  # progen can be supported but esm is preferred

        # Rename everything to self.language_model.transformer for convenience
        self.language_model.transformer = self.language_model.esm
        self.language_model.esm.embeddings.token_dropout = (
            False  # handles a bug in the ESM code when passing in embeddings
        )

    def add_offset_positions(self, cfg, batch):
        # Compute chain offsets
        hchain_offset = 0
        if cfg["condition_on_light_chain"] or cfg["predict_light_chain"]:
            lchain_id = cfg["lchain"]["id"]
            hchain_offset = len(batch["chain_seq_dict"][0][lchain_id])

        # Convert AA positions to tensors and compute offset_positions, which are positions + chain offsets
        for region in cfg["regions"].keys():
            input_positions = cfg["regions"][region]["positions"]
            if isinstance(input_positions, str):
                input_positions = eval(input_positions)
            input_positions = torch.LongTensor(input_positions)
            cfg["regions"][region]["positions"] = input_positions

            if cfg["regions"][region]["chain"] == "heavy":
                cfg["regions"][region]["offset_positions"] = (
                    cfg["regions"][region]["positions"] + hchain_offset
                )
            elif cfg["regions"][region]["chain"] == "light":
                cfg["regions"][region]["offset_positions"] = cfg["regions"][region][
                    "positions"
                ]

    def check_configs(self, cfg):
        assert (
            cfg["structure_enc_query"] ^ cfg["language_enc_query"]
        ), "Must pick exactly one of `structure_enc_query` or `language_enc_query`"
        if cfg["n_recycles"] >= 1:
            assert (
                cfg["recycle_greedy_decoded"]
                ^ cfg["recycle_logits"]
                ^ cfg["recycle_embeddings"]
            )
        assert not (
            cfg["linear_comb_embeddings"] and cfg["learn_linear_comb_embeddings"]
        )

    def fix_lm_vocab_order(self):
        assert self.model_type in ("progen", "esm"), "Did not recognize model type"
        all_embeddings = (
            self.get_language_model_embeddings()
        )  # vocab_size x lm_embed_dim
        token_to_lm_input_id = self.tokenizer._token_to_id

        # Hard-coded map for different tokens that have the same meaning across models
        rename_token_map = {
            "<cls>": "<s>",
            "<|bos|>": "<s>",
            "<eos>": "</s>",
            "<|eos|>": "<s>",
            "<|pad|>": "<pad>",
            "<|mask|>": "<mask>",
            "<|unk|>": "<unk>",
        }

        unmapped_tokens, lm_id_to_structure_id = list(), dict()
        for token, input_id in token_to_lm_input_id.items():
            if token in AA_TO_IDX:
                lm_id_to_structure_id[input_id] = AA_TO_IDX[token]
            elif token in rename_token_map:
                lm_id_to_structure_id[input_id] = AA_TO_IDX[rename_token_map[token]]
            else:
                unmapped_tokens.append(token)
        # map unknown tokens to garbage token
        for unmapped_token in unmapped_tokens:
            lm_id_to_structure_id[token_to_lm_input_id[unmapped_token]] = (
                len(AA_TO_IDX) - 1
            )
        remapping_idxs = torch.LongTensor(list(lm_id_to_structure_id.values()))

        self.set_language_model_embeddings(all_embeddings[remapping_idxs])

    def tokenize(self, sequence, **kwargs):
        encoded = self.tokenizer(sequence, **kwargs)
        if kwargs.get("add_special_tokens", False):
            input_ids = maybe_add_bos_eos_ids(encoded.input_ids, self.tokenizer)
            if self.model_type == "progen":
                backwards_input_ids = maybe_add_bos_eos_ids(
                    torch.flip(encoded.input_ids, [1]), self.tokenizer
                )
                encoded["input_ids"] = torch.cat(
                    (input_ids[:, :-1], backwards_input_ids), dim=1
                )
            else:
                encoded["input_ids"] = maybe_add_bos_eos_ids(
                    encoded.input_ids, self.tokenizer
                )
        return encoded

    # For getting linear combinations of input embeddings
    def coefs_to_lm_embeddings(self, embed_coefs):
        batch_size, seq_len, _ = embed_coefs.shape
        embed_coefs = embed_coefs.reshape((-1, embed_coefs.shape[-1]))
        all_embeddings = (
            self.get_language_model_embeddings()
        )  # vocab_size x lm_embed_dim
        _, lm_embed_dim = all_embeddings.shape
        reweighted_embeddings = torch.einsum(
            "n v, v d -> n d", embed_coefs, all_embeddings
        ).reshape((batch_size, seq_len, lm_embed_dim))
        return reweighted_embeddings

    # You must pass in the args. During training this is the same as cfg. Allows flexiblity during inference
    def refine_step(
        self,
        sequence: str | List[str] | List[List[str]],
        structure_node_feats,
        structure_logits,
        n_recycles,
        recycle_greedy_decoded,
        recycle_logits,
        recycle_embeddings,
    ) -> str | List[str] | List[List[str]]:

        lm_tokenized_sequence = self.tokenize(
            sequence,
            return_tensors="pt",
            add_special_tokens=self.cfg["use_special_tokens"],
            padding=True,
        ).to(self.device)

        for cycle in range(n_recycles + 1):
            attention_mask = lm_tokenized_sequence.attention_mask
            if self.cfg["use_special_tokens"] and self.model_type == "progen":
                attention_mask = torch.cat(
                    (
                        F.pad(attention_mask, (1, 0), value=1),
                        F.pad(attention_mask, (1, 1), value=1),
                    ),
                    dim=1,
                )
            if cycle == 0:
                if (
                    self.cfg["learn_linear_comb_embeddings"]
                    or self.cfg["linear_comb_embeddings"]
                ):
                    if self.cfg["linear_comb_embeddings"]:
                        probs = F.softmax(structure_logits, dim=-1)  # batch x seq x 32
                        default_threshold = torch.full(
                            (probs.shape[:2]),
                            self.cfg["linear_comb_embeddings_threshold"],
                        ).to(self.device)
                        backup_threshold = probs.max(dim=-1)[0].to(self.device)
                        threshold = torch.argmax(
                            torch.stack((default_threshold, backup_threshold), dim=-1),
                            dim=-1,
                        )
                        # Must use out-of-place implementation to allow gradient flow
                        torch.scatter(
                            input=probs,
                            dim=-1,
                            index=(probs < threshold[..., None]).long(),
                            src=torch.zeros_like(probs),
                        )
                        embed_coefs = probs
                        embed_coefs = probs / probs.sum(dim=-1, keepdims=True)
                        embed_coefs = F.pad(embed_coefs, (0, 1))
                    elif self.cfg["learn_linear_comb_embeddings"]:
                        embed_coefs = self.probs_project(
                            F.pad(F.softmax(structure_logits, dim=-1), (0, 1))
                        )
                    reweighted_embeddings = self.coefs_to_lm_embeddings(embed_coefs)

                    lm_outputs = self.language_model.transformer(
                        inputs_embeds=reweighted_embeddings,
                        attention_mask=attention_mask,
                    )
                else:
                    lm_outputs = self.language_model.transformer(
                        input_ids=lm_tokenized_sequence.input_ids,
                        attention_mask=attention_mask,
                    )
            elif recycle_greedy_decoded:
                lm_outputs = self.language_model.transformer(
                    input_ids=prediction_scores.argmax(-1),
                    attention_mask=attention_mask,
                )
            elif recycle_logits:
                torch.cuda.empty_cache()
                embed_coefs = self.probs_project(
                    F.softmax(prediction_scores, dim=-1)
                )  # batch x seq_len x vocab_size
                reweighted_embeddings = self.coefs_to_lm_embeddings(embed_coefs)
                lm_outputs = self.language_model.transformer(
                    inputs_embeds=reweighted_embeddings,
                    attention_mask=attention_mask,
                )

            elif recycle_embeddings:
                torch.cuda.empty_cache()
                lm_outputs = self.language_model.transformer(
                    inputs_embeds=sequence_output,
                    attention_mask=attention_mask,
                )

            sequence_output = lm_outputs[0]
            if self.cfg["structure_enc_query"]:
                query = structure_node_feats
                key, value = sequence_output, sequence_output
                if self.cfg["use_special_tokens"]:
                    query = F.pad(
                        input=query, pad=(0, 0, 1, 1)
                    )  # one zero on left and right
                if self.model_type == "progen" and self.cfg["use_special_tokens"]:
                    query = torch.cat((query[:, :-1], query), dim=1)
            elif self.cfg["language_enc_query"]:
                query = sequence_output
                if self.model_type == "progen" and self.cfg["use_special_tokens"]:
                    query = torch.cat((query[:, :-1], query), dim=1)
                key, value = structure_node_feats, structure_node_feats

            sequence_output = self.structural_adapter(query, key, value)

            if self.cfg["use_special_tokens"]:
                sequence_output = sequence_output[:, 1:-1, :]

            prediction_scores = self.language_model.lm_head(sequence_output)
            if self.model_type == "progen" and self.cfg["use_special_tokens"]:
                seq_len = structure_logits.shape[1]
                prediction_scores = prediction_scores[:, 1 : seq_len + 1, :]

            if self.cfg["structure_logits_skip_connection"]:
                if "esm" in self.cfg["pretrained_language_model_name_or_path"].lower():
                    prediction_scores += torch.nn.functional.pad(
                        structure_logits, (0, 1)
                    )
                else:
                    prediction_scores += structure_logits

        output = MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )
        return output

    def prevent_invalid_tokens_LM(self, logits):
        """Takes in logits from LM. Modifies them to only allow AAs and <pad>"""
        valid_tokens = [
            token for token in AA_TO_IDX.keys() if re.match("[A-Z]|<pad>", token)
        ]
        valid_token_ids = [
            AA_TO_IDX[token] for token in AA_TO_IDX.keys() if re.match("[A-Z]", token)
        ]
        if self.model_type == "esm":
            valid_mask = torch.zeros(len(self.tokenizer._id_to_token.keys())).bool()
        else:
            valid_mask = torch.zeros(len(self.tokenizer.vocab.keys())).bool()

        for valid_token_id in valid_token_ids:
            valid_mask[valid_token_id] = 1

        logits[:, :, ~valid_mask.bool()] = -np.inf
        return logits

    def get_language_model_embeddings(self):
        all_idxs = torch.LongTensor(range(self.language_model.config.vocab_size)).to(
            self.device
        )
        if self.model_type == "progen":
            all_embeddings = self.language_model.transformer.wte(all_idxs)
        elif self.model_type == "esm":
            all_embeddings = self.language_model.esm.embeddings.word_embeddings(
                all_idxs
            )
        else:
            raise RuntimeError("Did not recognize model type when getting embeddings")
        return all_embeddings

    def set_language_model_embeddings(self, new_embeddings):
        if self.model_type == "progen":
            self.language_model.transformer.wte.weight = nn.parameter.Parameter(
                new_embeddings
            )
        elif self.model_type == "esm":
            self.language_model.esm.embeddings.word_embeddings.weight = (
                nn.parameter.Parameter(new_embeddings)
            )
        else:
            raise RuntimeError("Did not recognize model type when getting embeddings")

    def CMLM_recycle(self, sequence, current_logits, batch):
        for recycle_count in range(self.cfg["n_CMLM_recycles"]):
            lm_tokenized_sequence = self.tokenize(
                sequence,
                return_tensors="pt",
                add_special_tokens=self.cfg["use_special_tokens"],
                padding=True,
            ).to(self.device)

            masking_fraction = 1.0 - (recycle_count + 1) / (
                self.cfg["n_CMLM_recycles"] + 1
            )
            num_masks = int(current_logits.shape[1] * masking_fraction)
            logit_entropy = Categorical(logits=current_logits).entropy()
            mask_locations = torch.topk(logit_entropy, num_masks, dim=1)[1]

            mask_token = self.tokenizer.mask_token_id
            lm_tokenized_sequence["input_ids"].scatter_(
                dim=1,
                index=mask_locations,
                src=torch.full_like(mask_locations, mask_token),
            )

            lm_outputs = self.language_model.transformer(
                input_ids=lm_tokenized_sequence.input_ids,
                attention_mask=torch.ones_like(lm_tokenized_sequence.input_ids),
            )

            current_logits = self.original_lm_head(lm_outputs[0])
        sequence = decode_logits_greedy(current_logits)
        decoded_sequences = self.fix_seq_string(sequence, batch)
        return sequence, decoded_sequences

    def fix_seq_string(self, sequence, batch):
        decoded_sequences = list()
        for pred_seq, seq in zip(sequence, batch["sequences"]):
            adjusted_pred_seq = "".join(pred_seq).replace(" ", "")[: len(seq)]
            decoded_sequences.append(adjusted_pred_seq)
            assert len(adjusted_pred_seq) == len(
                seq
            ), f"Predicted seq len ({len(pred_seq)})and seq len ({len(seq)}) did not match"
        return decoded_sequences

    def forward(self, batch, mode, T=None):
        """Runs the LM-Design model on an input batch"""
        batch["coords"] += self.cfg["noise"] * torch.rand_like(batch["coords"])
        structure_model_out = self.structure_model(batch)
        structure_logits = structure_model_out[0]["pred_final"]["sequence"]
        structure_node_feats = structure_model_out[0][
            "node_feats"
        ]  # node feats: (8, 438, 128)

        sequence = decode_logits_greedy(structure_logits)

        if np.random.random() < self.cfg["prob_ground_truth_seq"] and mode == "train":
            sequence = [
                "".join([IDX_TO_AA[input_id.item()] for input_id in tokenized_seq])
                for tokenized_seq in batch["tokenized_sequences"]
            ]

        if self.cfg["only_pmpnn"]:
            chain_lengths = (batch["chain_ids"] != -1).sum(axis=1)
            truncated_sequences = [
                seq[:length] for seq, length in zip(sequence, chain_lengths)
            ]
            structure_model_out[0]["pred_final"][
                "decoded_sequences"
            ] = truncated_sequences
            return structure_model_out

        should_recycle_greedy_decoded = (
            self.cfg["recycle_greedy_decoded"] if mode != "train" else False
        )
        output = self.refine_step(
            sequence,
            structure_node_feats,
            structure_logits,
            n_recycles=self.cfg["n_recycles"],
            recycle_greedy_decoded=should_recycle_greedy_decoded,
            recycle_logits=self.cfg["recycle_logits"],
            recycle_embeddings=self.cfg["recycle_embeddings"],
        )

        output_logits = self.prevent_invalid_tokens_LM(output.logits)
        sequence = decode_logits_greedy(output_logits)
        decoded_sequences = self.fix_seq_string(sequence, batch)

        if self.cfg["n_CMLM_recycles"]:
            sequence, decoded_sequences = self.CMLM_recycle(
                sequence, output_logits, batch
            )

        return (
            output | {"sequence": sequence} | {"decoded_sequences": decoded_sequences}
        )
