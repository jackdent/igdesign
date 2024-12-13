import cytoolz as ct
import torch
from hydra.utils import instantiate
from torch import nn

from igdesign.structure_utils import masked_align_structures
from igdesign.tokenization import detokenize, AA_TO_IDX, IDX_TO_AA
from igdesign.model import Model
from igdesign.models.lmdesign_model import LMDesignModel
from igdesign.embedding import InputEmbedding


@torch.no_grad()
def weight_reset(m: nn.Module):
    # Check if the current module has reset_parameters and if it's called on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        print("Resetting parameters.")
        m.reset_parameters()


class IFWrapper(Model):
    def __init__(
        self, cfg, res_dim_hidden: int = 128, pair_dim_hidden: int = 128, **ignore
    ):
        super().__init__(cfg, **ignore)
        self.cfg = cfg
        self.all_supervised = getattr(cfg.model, "all_supervised", [])
        self.res_dim_hidden, self.pair_dim_hidden = res_dim_hidden, pair_dim_hidden
        self.to_logits = nn.Sequential(
            nn.LayerNorm(res_dim_hidden), nn.Linear(res_dim_hidden, len(AA_TO_IDX))
        )
        self.embedding = InputEmbedding(
            self.feature_factory,
            res_embed_dim=res_dim_hidden,
            pair_embed_dim=pair_dim_hidden,
        )
        self.coord_noise = cfg.model.coord_noise
        self.inference = False
        if getattr(self.config.model, "re_init_logit_proj", False):
            self.to_logits.apply(weight_reset)

    def mask_out_residue_features(self, batch):
        batch["features"]["res_ty"].raw_data *= 0
        batch["features"]["res_ty"].encoded_data *= 0

    def forward(self, batch, return_feats=False, *ignore_args, **ignore_kwargs):
        self.mask_out_residue_features(batch)
        node_feats, pair_feats = self.embedding(batch["features"])
        res_out, pair_out, coord_out = self.model(
            node_feats=node_feats,
            pair_feats=pair_feats,
            **self.model.get_forward_kwargs_from_batch(batch)
        )
        return self.postprocess(batch, res_out, pair_out, coord_out)

    def init_model(self, model_config, **ignore):
        self.model = instantiate(model_config.if_model)
        if "freeze_if_weights" in model_config:
            if model_config.freeze_if_weights:
                print("[INFO] Freezing inverse folder weights")
                for _, param in self.model.named_parameters():
                    param.requires_grad = False

    def collate(self, batch, precollate):
        sequences = [detokenize(x["tokenized_sequences"]) for x in batch]
        for item in batch:
            item["tokenized_sequences"][
                ~item["masks"]["valid"]["residue"]
            ] = 25  # mask token ("X")
        batch = precollate(batch)
        batch_copy = {k: v for k, v in batch.items()}
        if self.coord_noise and self.training:
            noise = torch.randn_like(batch["coords"]) * self.coord_noise
            batch_copy["coords"] = batch["coords"] + noise
            batch["denoised_coords"] = batch["coords"]
            batch["coords"] = batch_copy["coords"]
        batch["features"] = self.feature_factory.generate(batch_copy)

        batch["masks"]["all_supervised"] = dict()
        batch["masks"]["all_supervised"]["residue"] = torch.zeros_like(
            batch["masks"]["all"]["residue"]
        )
        batch["masks"]["all_supervised"]["atom"] = torch.zeros_like(
            batch["masks"]["all"]["atom"]
        )
        for mask_name in self.all_supervised:
            batch["masks"]["all_supervised"]["residue"] += batch["masks"][mask_name][
                "residue"
            ]
            batch["masks"]["all_supervised"]["atom"] += batch["masks"][mask_name][
                "atom"
            ]

        batch["sequences"] = sequences
        return batch

    def subset_masks_to_valid(self, batch):
        res_mask, atom_mask = ct.get(["residue", "atom"], batch["masks"]["valid"])
        mask_dict = {}
        for k, v in batch["masks"].items():
            mask_dict[k] = {}
            if "residue" in v:
                mask_dict[k]["residue"] = v["residue"] & res_mask
            if "atom" in v:
                mask_dict[k]["atom"] = v["atom"] & atom_mask
        return mask_dict

    def postprocess(self, batch, res_out, pair_out, coord_out, logits=None):
        mask_dict = self.subset_masks_to_valid(batch)
        valid_atom = mask_dict["valid"]["atom"]
        mask_dict["pred_final"] = mask_dict["valid"]

        chain_aligned_coords = (
            masked_align_structures(
                coord_out,
                batch["coords"],
                valid_atom,
            )
            if coord_out is not None
            else None
        )

        output_logits = logits if logits is not None else self.to_logits(res_out)
        decoded_sequences = list()
        token_ids = output_logits.argmax(dim=-1)
        for batch_idx in range(len(token_ids)):
            decoded_sequences.append(
                "".join(
                    [IDX_TO_AA[t] for t in token_ids[batch_idx].cpu().numpy().tolist()]
                )
            )

        results = dict(
            chain_aligned=dict(coordinates=chain_aligned_coords),
            pred_final=dict(
                coordinates=coord_out,
                sequence=output_logits,
                decoded_sequences=decoded_sequences,
            ),
        )

        results["node_feats"] = res_out
        results["pair_feats"] = pair_out
        return results, mask_dict


class LMDesignIFWrapper(IFWrapper):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        super().__init__(cfg, **kwargs)

    def load_protein_mpnn(self):
        model = IFWrapper.load_from_checkpoint(self.pmpnn_path)
        return model

    def init_model(self, model_cfg, **kwargs):
        model = self.load_protein_mpnn()
        self.model = LMDesignModel(dict(model_cfg), structure_model=model, **kwargs)

    def fix_tokenized_sequences(self, tokenized_sequences):
        """
        Seqs differ in tokenization, so we need to re-encode tokenized seqs to match LM logits.
        """
        decoded = self.model.structure_model.batch_decode(tokenized_sequences)
        lm_encoded = self.model.language_model.tokenizer(decoded, return_tensors="pt")[
            "input_ids"
        ]
        return lm_encoded

    def forward(self, batch, *args, **kwargs):
        self.mask_out_residue_features(batch)
        model_out = self.model.forward(
            batch, mode=kwargs["mode"]
        )  # lmd fwd for seqs/refinement
        results, mask_dict = self.postprocess(batch, None, None, None, model_out)
        return results, mask_dict

    def postprocess(self, batch, res_out, pair_out, coord_out, model_out):
        mask_dict = self.subset_masks_to_valid(batch)
        valid_atom = mask_dict["valid"]["atom"]

        mask_dict["pred_final"] = mask_dict["valid"]

        chain_aligned_coords = (
            masked_align_structures(
                coord_out,
                batch["coords"],
                valid_atom,
            )
            if coord_out is not None
            else None
        )

        if self.cfg["model"]["only_pmpnn"]:
            return model_out[0], mask_dict
        else:
            output_logits = model_out.logits

        decoded_sequences = list()
        token_ids = output_logits.argmax(dim=-1)
        for batch_idx in range(len(token_ids)):
            decoded_sequences.append(
                "".join(
                    [IDX_TO_AA[t] for t in token_ids[batch_idx].cpu().numpy().tolist()]
                )
            )

        results = dict(
            chain_aligned=dict(coordinates=chain_aligned_coords),
            pred_final=dict(
                coordinates=coord_out,
                sequence=output_logits,
                decoded_sequences=decoded_sequences,
            ),
        )

        results["node_feats"] = res_out
        results["pair_feats"] = pair_out
        return results, mask_dict
