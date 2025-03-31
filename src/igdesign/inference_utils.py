import logging
import re
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from einops import rearrange, repeat

from igdesign.tokenization import AA_TO_IDX, IDX_TO_AA

logger = logging.getLogger(__name__)

nat_aas = "ARNDQEGHILKMFPSTWYV"  # C - removing CYS # X - added unknown X
nat_aa_posns = [AA_TO_IDX[s] for s in nat_aas]
nat_aa_mask = torch.zeros(len(AA_TO_IDX))
nat_aa_mask[torch.tensor(nat_aa_posns)] = 1
nat_aa_mask = nat_aa_mask.bool()
non_nat_aa_idxs = torch.arange(len(AA_TO_IDX))[~nat_aa_mask]


def aa_to_tensor(aa_list):
    """Recursively convert a list of amino acid characters into a LongTensor"""
    if isinstance(aa_list[0], str):
        return torch.LongTensor(
            [[AA_TO_IDX[aa] for aa in aa_seq] for aa_seq in aa_list]
        )
    return torch.stack([aa_to_tensor(sublist) for sublist in aa_list])


def tensor_to_aa(input_tensor):
    return "".join([IDX_TO_AA[idx.item()] for idx in input_tensor])


def score_single_sequence(logits, positions, sequence, AA_TO_IDX):
    """Compute cross entropy score for a single sequence against N logits."""
    seq_to_score = sequence[positions]
    seq_to_score = repeat(seq_to_score, "l -> b l", b=logits.shape[0])
    logits_for_scoring = logits[:, positions, :]

    logits_for_scoring = rearrange(logits_for_scoring, "b l c -> b c l")
    losses = cross_entropy(logits_for_scoring, seq_to_score, reduction="none")
    losses = losses.mean(dim=1)
    return losses


def shuffle_tensor(input_tensor):
    """Randomly shuffles a tensor along its first dimension."""
    shuffled_idxs = torch.randperm(input_tensor.shape[0])
    return input_tensor[shuffled_idxs]


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

    # Make batch_size decoding orders.
    # Shuffle the non-design positions, then each of the design regions in order.
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


def build_temperature_arr(cfg, batch):
    """Build a 1D array of temperatures corresponding to locations in the sequence."""
    temperatures = torch.full((batch["tokenized_sequences"].shape[1],), np.nan)
    for region in cfg["regions"].keys():
        region_positions = cfg["regions"][region]["offset_positions"]
        temperature = cfg["regions"][region]["temperature"]
        temperatures[region_positions] = temperature
    return temperatures


def get_independent_logits(batch, cfg, model):
    """
    Compute logits for each residue while conditioning only on the structure and framework sequence.
    """
    combined_running_logits = list()
    for _ in range(int(cfg["num_batches"])):
        # get decode order
        decode_order, offset, num_design_positions = get_decode_order(cfg, batch)
        decode_order = decode_order.to(model.device)
        batch["decode_order"] = decode_order
        results, *_other = model(batch)
        running_logits = results["pred_final"]["sequence"]

        design_decoding_positions = torch.cat(
            [
                cfg["regions"][region]["offset_positions"]
                for region in cfg["region_order"]
            ]
        )
        for aa_position in design_decoding_positions:
            decode_order[:, offset] = aa_position
            batch["decode_order"] = decode_order
            results, *_other = model(batch)
            logits = results["pred_final"]["sequence"]
            running_logits[:, aa_position] = logits[:, aa_position]
        combined_running_logits.append(running_logits)
    return torch.cat(combined_running_logits, axis=0)


def compute_independent_loss(cfg, seqs, batch, model, decode_order : torch.Tensor | None = None):
    independent_losses = {
        "independent_" + region_name: list() for region_name in cfg["regions"].keys()
    }
    independent_logits = get_lmdesign_logits(batch, cfg, model, decode_order = decode_order)
    for seq in tqdm(seqs, desc="Scoring sequences with independent loss"):
        for region_name in cfg["regions"].keys():
            region_positions = cfg["regions"][region_name]["offset_positions"]
            ce_losses = score_single_sequence(
                independent_logits, region_positions, seq, AA_TO_IDX
            )
            independent_losses["independent_" + region_name].append(ce_losses)
    return independent_losses


def prevent_invalid_tokens(logits):
    valid_idxs = [
        idx for token, idx in AA_TO_IDX.items() if re.match("[A-Z]|<pad>", token)
    ]
    invalid_idxs = [idx for idx in range(logits.shape[-1]) if idx not in valid_idxs]
    logits[:, :, invalid_idxs] = -1e10
    return logits


def get_lmdesign_logits(batch, cfg, model, decode_order : torch.Tensor | None = None):

    if decode_order is None:
        decode_order, offset, num_design_positions = get_decode_order(cfg, batch)
        batch["decode_order"] = decode_order.to(model.device)
    else:
        logger.warning("Using precomputed decode order")
        batch["decode_order"] = decode_order.to(model.device)
       

    print(batch["decode_order"])

    # Rearrange decoding order here
    structure_model_out = model.model.structure_model(batch)
    structure_node_feats = structure_model_out[0]["node_feats"]
    structure_logits = structure_model_out[0]["pred_final"]["sequence"]

    structure_logits = prevent_invalid_tokens(structure_logits)

    pmpnn_sequence = decode_logits_greedy(structure_logits)

    output = model.model.refine_step(
        pmpnn_sequence,
        structure_node_feats,
        structure_logits,
        n_recycles=0,
        recycle_greedy_decoded=False,
        recycle_logits=False,
        recycle_embeddings=False,
    )

    output_logits = prevent_invalid_tokens(output.logits)
    return output_logits


def lmdesign_sample(batch, cfg, model):
    NUM_DECODING_ORDERS = cfg["lmdesign_num_decoding_orders"]
    NUM_PMPNN_SEQS = cfg["lmdesign_num_pmpnn_seqs"]
    NUM_LM_SEQS = cfg["lmdesign_num_lm_seqs"]
    PMPNN_LOGIT_TEMPERATURE = cfg["lmdesign_pmpnn_logit_temperature"]
    LM_LOGIT_TEMPERATURE = cfg["lmdesign_output_logit_temperature"]

    all_seqs, all_pmpnn_seqs = list(), list()
    for decoding_order_idx in range(NUM_DECODING_ORDERS):
        decode_order, offset, num_design_positions = get_decode_order(cfg, batch)
        batch["decode_order"] = decode_order.to(model.device)

        # Rearrange decoding order here
        structure_model_out = model.model.structure_model(batch)
        structure_node_feats = structure_model_out[0]["node_feats"]
        structure_logits = structure_model_out[0]["pred_final"]["sequence"]

        structure_logits = prevent_invalid_tokens(structure_logits)

        pmpnn_samples = torch.distributions.categorical.Categorical(
            logits=structure_logits[0] / PMPNN_LOGIT_TEMPERATURE
        ).sample((NUM_PMPNN_SEQS,))
        pmpnn_samples = batch_decode(pmpnn_samples)
        all_pmpnn_seqs.extend(pmpnn_samples)

        num_unique_pmpnn_samples = len(set(pmpnn_samples))

        for sequence in pmpnn_samples:
            output = model.model.refine_step(
                sequence,
                structure_node_feats,
                structure_logits,
                n_recycles=0,
                recycle_greedy_decoded=False,
                recycle_logits=False,
                recycle_embeddings=False,
            )

            output_logits = prevent_invalid_tokens(output.logits)
            lm_samples = torch.distributions.categorical.Categorical(
                logits=output_logits[0] / LM_LOGIT_TEMPERATURE
            ).sample((NUM_LM_SEQS,))
            all_seqs.extend(lm_samples)
    all_seqs = torch.stack(all_seqs)
    return all_seqs


def sample_sequences(df, batch, cfg, model):
    """Samples are conditioned only on model and structure."""
    seqs = lmdesign_sample(batch, cfg, model)
    return seqs, df


@torch.no_grad()
def sample(model, cfg, batch, save_root=None):
    """
    Run inference.

    1) Generate sequences by sampling from the model.
    2) Generate logits, depending on which losses you specify in the configs.
    3) Score the sequences against the logits using cross entropy loss.
    4) Return dataframe with sequences and scores.
    """
    df = pd.DataFrame(index=range(1))
    seqs, df = sample_sequences(df, batch, cfg, model)
    all_losses = dict()

    if cfg["independent_loss"]:
        independent_losses = compute_independent_loss(cfg, seqs, batch, model)
        all_losses.update(independent_losses)

    # Stack losses into a single tensor so we can convert to dataframe
    all_losses = {key: torch.stack(value) for key, value in all_losses.items()}
    combined_seqs = list()
    for seq_idx, seq in enumerate(seqs):
        combined_str = ""
        for region_idx, region_name in enumerate(cfg["region_order"]):
            region_subset = (
                seq[cfg["regions"][region_name]["offset_positions"]].cpu().numpy()
            )
            combined_str += tensor_to_aa(region_subset)
            if region_idx != len(cfg["region_order"]) - 1:
                combined_str += "|"  # region delimiter
        combined_seqs.append(combined_str)
    df = pd.concat((df, pd.DataFrame(combined_seqs, columns=["sampled_seq"])), axis=1)
    for col in df.columns:
        if col != "sampled_seq":
            df[col] = df[col].iloc[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for loss_type in all_losses.keys():  # all_losses[loss_type]: n_seqs x n_losses
            column_name = [f"ce_loss_{loss_type}"]
            all_loss_values = list()
            for seq_idx in range(seqs.shape[0]):
                loss_values = all_losses[loss_type][seq_idx].cpu().numpy().tolist()
                all_loss_values.append(
                    "|".join([str(round(value, 4)) for value in loss_values])
                )
            df.loc[:, column_name] = all_loss_values

    return df