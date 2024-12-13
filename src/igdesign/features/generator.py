from __future__ import annotations
import math
from typing import Optional, List, Union, Tuple, Any, Dict
from torchtyping import TensorType
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from einops import rearrange
import numpy as np

from igdesign.tokenization import AA_TO_IDX
from igdesign.features.feature_utils import (
    default,
    exists,
    get_bb_dihedral,
    get_tr_rosetta_orientation_mats,
    bin_encode,
    pair_mask_from_res_mask,
    valid_dihedral_mask,
    impute_beta_carbon,
    get_inter_chain_contacts,
)
from igdesign.features.feature_constants import (
    FeatureEncodingTy,
    FeatureID,
    FeatureTy,
    FeatureDescription,
    DEFAULT_CENTRALITY_RADII,
    DEFAULT_PW_DIST_RADII,
    SMALL_SEP_BINS,
    NATURAL_AA_INDICES,
    ATOM_TY_TO_ATOM_IDX,
    ENCODING_TYS,
    FEATURE_TYS,
    FEATURE_NAMES,
    RES_FEATURE_TYS,
    DEFAULT_COORD_IDX,
)
from igdesign.features.feature import Feature

GenerateType = Union[Tuple[Tensor, Any], Tuple[Tensor, Any, Optional[Dict[str, Any]]]]


class FeatureGenerator(ABC):
    def __init__(self, description: FeatureDescription):
        self.description = description

    @property
    def encoding_ty(self) -> FeatureEncodingTy:
        assert (
            self.description.encoding_ty in ENCODING_TYS
        ), f"unrecognized encoding ty {self.description.encoding_ty} for {self}"
        return self.description.encoding_ty

    @property
    def feature_ty(self) -> FeatureTy:
        assert (
            self.description.feature_ty in FEATURE_TYS
        ), f"unrecognized feature ty {self.description.feature_ty} for {self}"
        return self.description.feature_ty

    @property
    def feature_name(self) -> str:
        assert (
            self.description.feature_name in FEATURE_NAMES
        ), f"unrecognized name: {self.description.feature_name} for {self}"
        return self.description.feature_name

    @abstractmethod
    def can_mask(self) -> bool:
        """Return whether or not this feature can be masked"""
        pass

    def get_mask_value(self) -> Tensor:
        """Get value used to mask this feature"""
        pass

    """Default Masking Behavior -- Can be overridden in implementing class"""

    def _apply_mask(self, feature: Feature, mask: Tensor) -> Feature:
        feature.encoded_data[mask] = self.get_mask_value().to(
            device=feature.device, dtype=feature.dtype
        )
        return feature

    def apply_mask(self, feature: Feature, mask: Tensor) -> Feature:
        assert self.can_mask(), f"feature : {feature} can't be masked!"
        assert (
            mask.shape == feature.leading_shape
        ), f"mask shape : {mask.shape}, feature leading shape {feature.leading_shape}, feature: {feature}"
        return self._apply_mask(feature, mask)

    """Generation"""

    @abstractmethod
    def _generate(batch: Dict[str, Any]) -> GenerateType:
        """
        Generate the feature defined in subclass.

        Called by superclass generate function

        Args:
            batch (Dict[str, Any]): batch used to generate the feature

        Returns:
            encoded_feature: the (intermediate) encoding for the feature defined
            by the subclass.
            raw_feature: raw values for unencoded feature.

         NOTE: Tensors should be returned (as tuple) in order (encoded, raw)
        """
        pass

    def generate(self, batch: Dict[str, Any], **kwargs) -> Feature:
        feature = Feature(self.description, *self._generate(batch, **kwargs))
        expected_shape = 2 if self.description.feature_ty in RES_FEATURE_TYS else 3
        assert len(feature.leading_shape) == expected_shape, print(
            f"len(feature.leading_shape) = {len(feature.leading_shape)}, expected_shape = {expected_shape}"
        )
        return feature

    def __repr__(self):
        return (
            f"FeatureGenerator : {self.feature_name}-"
            f"{self.feature_ty.value}, EncodingTy : {self.encoding_ty.value}"
        )

    def __getattr__(self, attr):
        """Called only if this class does not have the given attribute"""
        try:
            return getattr(self.description, attr)
        except:
            raise AttributeError(f"No attribute {attr} found for this class")


class ResidueType(FeatureGenerator):
    def __init__(
        self,
        one_hot: bool = False,
        embed: bool = False,
        embed_dim: int = None,
        lm_embed: bool = False,
        pretrained_model_name_or_path: str = None,
        use_pretrained_weights: bool = True,
        freeze_lm: bool = True,
        use_cache: bool = True,
        cache_dir: str = "./",
        cache_clear_dir: bool = False,
        cache_verbose: bool = False,
        corrupt_prob: float = 0,
    ):
        """
        Residue Type Feature.

        Args:
            one_hot (bool, optional): use a one-hot encoding of residue type. Defaults to False.
            embed (bool, optional): embed the residue type with nn.Embedding. Defaults to False.
            embed_dim (int, optional): output dimension of nn.Embedding (if specified). Defaults to None.
            corrupt_prob (float, optional): probability with which to corrupt the amino acid type.
            If >0, each amino acid in the sequence will be replaced with one of the 20 standard amino
            acids with the given probability before encoding. Defaults to 0.
        """
        assert (
            one_hot ^ embed ^ lm_embed
        ), f"one-hot or embed or lm_embed must be specified!"

        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.RES_TY,
                encoding_ty=(
                    FeatureEncodingTy.ONEHOT
                    if one_hot
                    else (
                        FeatureEncodingTy.EMBED if embed else FeatureEncodingTy.LM_EMBED
                    )
                ),
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                use_pretrained_weights=use_pretrained_weights,
                freeze_lm=freeze_lm,
                use_cache=use_cache,
                cache_dir=cache_dir,
                cache_clear_dir=cache_clear_dir,
                cache_verbose=cache_verbose,
                embed_dim=embed_dim,
                num_classes=len(AA_TO_IDX),
            )
        )
        self.corrupt_prob = corrupt_prob

    def can_mask(self) -> bool:
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_value(self) -> TensorType[1]:
        """Get value used to mask this feature"""
        # Mask is the last index in the vocabulary
        mask_index = len(AA_TO_IDX)
        return torch.tensor([mask_index])

    def _corrupt_seq(
        self, batch: Dict[str, Any], seq_emb: TensorType["batch", "seq", "dim"]
    ) -> Tensor:
        """Corrupt the sequence with the given probability"""
        corrupt_mask = (torch.rand_like(seq_emb.float()) < self.corrupt_prob) & batch[
            "masks"
        ]["valid"]["residue"]
        corrupt_aas = torch.randint(
            0, len(NATURAL_AA_INDICES), size=(len(corrupt_mask[corrupt_mask]),)
        )
        replace_aas = torch.tensor(NATURAL_AA_INDICES)[corrupt_aas]
        seq_emb[corrupt_mask] = replace_aas
        return seq_emb

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        seq_emb = batch["tokenized_sequences"].long().clone()
        if self.corrupt_prob > 0:
            seq_emb = self._corrupt_seq(batch, seq_emb)
        return (seq_emb.unsqueeze(-1), batch["tokenized_sequences"].long().clone())


class BackboneDihedral(FeatureGenerator):
    def __init__(
        self,
        one_hot: bool = False,
        fourier: bool = False,
        num_classes: int = 36,
        num_fourier_feats: int = 2,
    ):
        """
        Backbone Dihedral
        Args:
            one_hot (bool, optional): _description_. Defaults to False.
            fourier (bool, optional): _description_. Defaults to False.
            encode_bins (int, optional): _description_. Defaults to 36.
            num_fourier_feats (int, optional): _description_. Defaults to 2.
        """
        assert one_hot ^ fourier, f"one-hot or fourier encoding must be specified!"
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.BB_DIHEDRAL,
                encoding_ty=(
                    FeatureEncodingTy.ONEHOT if one_hot else FeatureEncodingTy.FOURIER
                ),
                num_classes=num_classes + 2,  # extra bins for nan-values and mask value
                num_fourier_feats=num_fourier_feats,
                mult=3,
            )
        )
        self.num_classes, self.fourier_feats = num_classes, num_fourier_feats

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return self.encoding_ty == FeatureEncodingTy.ONEHOT

    def get_mask_value(self) -> TensorType[1]:
        """Get value used to mask this feature"""
        return torch.tensor([self.num_classes + 1])

    def _apply_mask(self, feature, mask: Tensor) -> Feature:
        """Mask the feature"""
        # We pass the full input_mask + valid_res_mask, because both affect whether a dihedral
        # is valid in a given position
        mask_val = self.get_mask_value().to(device=feature.device, dtype=feature.dtype)
        valid_dihedrals_mask = valid_dihedral_mask(
            valid_res_mask=~mask, res_ids=feature.res_ids
        )
        feature.encoded_data[~valid_dihedrals_mask] = mask_val
        return feature

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        b, n = batch["coords"].shape[:2]
        N, CA, C, *_ = batch["coords"].unbind(dim=-2)
        bb_dihedrals = torch.cat(
            [x.unsqueeze(-1) for x in get_bb_dihedral(N, CA, C)], dim=-1
        )

        nan_mask = torch.isnan(bb_dihedrals)
        encoded_feats = bb_dihedrals.clone()

        if self.encoding_ty == FeatureEncodingTy.ONEHOT:
            encoded_bb_dihedrals = torch.clamp(((bb_dihedrals / math.pi) + 1) / 2, 0, 1)
            encoded_feats = encoded_bb_dihedrals * (self.num_classes - 1)
            encoded_feats[nan_mask] = self.num_classes

        bb_dihedrals[nan_mask] = 0

        return encoded_feats, bb_dihedrals, dict(res_ids=batch["res_ids"])


class TrRosettaOrientation(FeatureGenerator):
    def __init__(
        self,
        one_hot: bool = False,
        fourier: bool = False,
        rbf: bool = False,
        num_classes: int = 36,
        num_fourier_feats: int = 2,
        rbf_radii: Optional[List] = None,
        rbf_sigma: float = 0.15,
    ):
        """
        Tr-Rosetta phi,psi, and omega Orientation
        (see: https://www.nature.com/articles/s41596-021-00628-9)

        Args:
            one_hot (bool, optional): _description_. Defaults to False.
            fourier (bool, optional): _description_. Defaults to False.
            encode_bins (int, optional): _description_. Defaults to 36.
            num_fourier_feats (int, optional): _description_. Defaults to 2.
        """

        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.TR_ORI,
                encoding_ty=(
                    FeatureEncodingTy.ONEHOT
                    if one_hot
                    else (
                        FeatureEncodingTy.FOURIER if fourier else FeatureEncodingTy.RBF
                    )
                ),
                num_classes=num_classes + 2,  # extra bins for nan and mask
                num_fourier_feats=num_fourier_feats,
                mult=int(3 * (1 + int(rbf))),
                rbf_radii=default(
                    torch.linspace(start=-1, end=1, steps=int(2 / rbf_sigma)), rbf_radii
                ),
                rbf_sigma=rbf_sigma,
            )
        )
        self.num_classes, self.fourier_feats = num_classes, num_fourier_feats

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return self.encoding_ty != FeatureEncodingTy.FOURIER

    def get_mask_value(self) -> Tensor:
        """Get value used to mask this feature"""
        val = (
            -1e5
            if self.encoding_ty == FeatureEncodingTy.RBF
            else (self.num_classes + 1)
        )
        return torch.tensor([val])

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        N, CA = map(
            lambda atom: batch["coords"][:, :, ATOM_TY_TO_ATOM_IDX[atom]], ("N", "CA")
        )
        CB = impute_beta_carbon(batch["coords"], batch["masks"]["valid"]["residue"])
        phi, psi, omega = get_tr_rosetta_orientation_mats(N, CA, CB)
        ori_feats = torch.cat([x.unsqueeze(-1) for x in (phi, psi, omega)], dim=-1)
        encoded_feats = ori_feats.clone()
        encoded_feats[torch.isnan(ori_feats)] = 0.0
        if self.encoding_ty in [FeatureEncodingTy.ONEHOT, FeatureEncodingTy.EMBED]:
            encoded_feats = torch.clamp(((ori_feats / math.pi) + 1) / 2, 0, 1) * (
                self.num_classes - 1
            )
            encoded_feats[torch.isnan(ori_feats)] = self.num_classes
        elif self.encoding_ty == FeatureEncodingTy.RBF:
            sin_feats, cos_feats = torch.sin(ori_feats), torch.cos(ori_feats)
            sin_feats[torch.isnan(ori_feats)] = 1e5
            cos_feats[torch.isnan(ori_feats)] = 1e5
            encoded_feats = torch.cat(sin_feats, cos_feats, dim=-1)
        else:
            ori_feats[torch.isnan(ori_feats)] = 0  #:(

        return encoded_feats, ori_feats


class ResidueDegreeCentrality(FeatureGenerator):
    def __init__(
        self,
        one_hot: bool = False,
        rbf: bool = False,
        num_classes: int = 5,
        min_value: int = 6,
        max_value: int = 40,
        radius: float = 10,
        rbf_radii: Optional[List] = None,
        rbf_sigma: float = 4.0,
    ):
        """
        Residue Degree Centrality Feature

        Counts the number of CB atoms within a `radius` angstrom ball
        around a query residues CB atom (CA is used if CB is not available).

        Args:
            one_hot one-hot encode this feature. Defaults to False.
            rbf (bool, optional): RBF encode this feature. Defaults to False.
            encode_bins (int, optional): number of (equal-width) bins to encode to. Defaults to 8.
            min_value (int, optional): _description_. Defaults to 6.
            max_value (int, optional): _description_. Defaults to 40.
            radius (float, optional): _description_. Defaults to 12.
            rbf_radii (Optional[List], optional): _description_. Defaults to None.
            rbf_sigma (float, optional): _description_. Defaults to 4.0.
        """
        assert one_hot ^ rbf, f"one-hot or fourier encoding must be specified!"
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.CENTRALITY,
                encoding_ty=(
                    FeatureEncodingTy.ONEHOT if one_hot else FeatureEncodingTy.RBF
                ),
                num_classes=num_classes + 1,
                rbf_radii=rbf_radii,
                rbf_sigma=rbf_sigma,
            )
        )
        self.num_classes = num_classes
        self.bounds = (min_value, max_value)
        self.radius = radius
        self.rbf_radii = default(rbf_radii, DEFAULT_CENTRALITY_RADII)
        self.rbf_sigma = rbf_sigma

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_value(self) -> Tensor:
        """Get value used to mask this feature"""
        val = -1e5 if self.encoding_ty == FeatureEncodingTy.RBF else (self.num_classes)
        return torch.tensor([val])

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        coords = batch["coords"][:, :, DEFAULT_COORD_IDX]
        mask = pair_mask_from_res_mask(batch["masks"]["valid"]["residue"])
        dists = torch.cdist(coords, coords)
        dists[~mask.bool()] = self.radius + 1  # mask out missing coordinates
        feat = torch.sum(dists <= self.radius, dim=-1).float() - 1  # fill in for chain

        encoded_feat = feat.clone()
        if self.encoding_ty == FeatureEncodingTy.ONEHOT:
            cmin, cmax = self.bounds
            clamped_centrality = torch.clamp(encoded_feat, cmin, cmax) - cmin
            norm_res_centrality = clamped_centrality / (cmax - cmin)
            encoded_feat = norm_res_centrality.float() * (self.num_classes - 1)
        return map(lambda x: x.unsqueeze(-1), (encoded_feat, feat))


class DiscreteEncoding(FeatureGenerator):
    def __init__(
        self,
        feature_id,
        embed_dim: int = 24,
        num_classes: int = 200,
    ):
        super().__init__(
            FeatureDescription(
                feature_id=feature_id,
                encoding_ty=FeatureEncodingTy.EMBED,
                embed_dim=embed_dim,
                num_classes=num_classes,
            )
        )
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def can_mask(self):
        return False

    def get_mask_value(self) -> Tensor:
        raise Exception("Discrete encoding can't be masked!")


class TimestepEncoding(DiscreteEncoding):
    def __init__(
        self,
        embed_dim: int = 24,
        num_classes: int = 200,
    ):
        super().__init__(FeatureID.TIMESTEP_ENC, embed_dim, num_classes)

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        timesteps = torch.full_like(
            batch["res_ids"], fill_value=batch["timestep"]
        ).unsqueeze(-1)
        return timesteps, timesteps


class TaskEncoding(DiscreteEncoding):
    def __init__(
        self,
        embed_dim: int = 24,
        num_classes: int = 200,
    ):
        super().__init__(FeatureID.TASK_ENC, embed_dim, num_classes)

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        tasks = torch.full_like(
            batch["res_ids"], fill_value=batch["task_index"]
        ).unsqueeze(-1)
        return tasks, tasks


class PositionalEncoding(FeatureGenerator):
    def __init__(
        self,
        fourier: bool = False,
        embed: bool = False,
        num_fourier_feats: int = 12,
        embed_dim: int = 24,
    ):
        assert fourier ^ embed, f"fourier or embed must be specified!"
        encoding_ty = FeatureEncodingTy.FOURIER if fourier else FeatureEncodingTy.EMBED
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.POS_ENC,
                encoding_ty=encoding_ty,
                num_fourier_feats=num_fourier_feats,
                embed_dim=embed_dim,
            )
        )
        self.fourier_feats = num_fourier_feats
        self.embed_dim = embed_dim

    def can_mask(self):
        return False

    def get_mask_value(self) -> Tensor:
        raise Exception("Positional encoding can't be masked!")

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        res_posns = batch["res_ids"]
        return res_posns.unsqueeze(-1), res_posns.unsqueeze(-1)


class RelativeSequenceSeparation(FeatureGenerator):
    def __init__(
        self,
        one_hot: bool = False,
        fourier: bool = False,
        embed: bool = False,
        sep_bins: Optional[List[int]] = None,
        num_fourier_feats: int = 12,
        embed_dim: int = 24,
    ):
        """
        Relative Sequence Separation Encoding

        Args:
            one_hot (bool, optional): _description_. Defaults to False.
            fourier (bool, optional): _description_. Defaults to False.
            embed (bool, optional): _description_. Defaults to False.
            sep_bins (List,optional): list of bin edges to use for one-hot embedding.
             Defaults to features.constants.SMALL_SEP_BINS.
            fourier_feats (int, optional): _description_. Defaults to 12.
            embed_dim (int, optional): _description_. Defaults to 24.
        """
        assert (
            one_hot ^ fourier ^ embed
        ), f"one-hot or fourier encoding must be specified!"
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.REL_SEP,
                encoding_ty=(
                    FeatureEncodingTy.ONEHOT
                    if one_hot
                    else (
                        FeatureEncodingTy.FOURIER
                        if fourier
                        else FeatureEncodingTy.EMBED
                    )
                ),
                num_fourier_feats=num_fourier_feats,
                num_classes=len(default(sep_bins, SMALL_SEP_BINS)) + 1,
                embed_dim=embed_dim,
            )
        )
        self.fourier_feats = num_fourier_feats
        self.embed_dim = embed_dim
        self.sep_bins = default(sep_bins, SMALL_SEP_BINS)
        self.num_classes = len(self.sep_bins)

    def can_mask(self):
        # We choose to not let this feature be masked by default, because
        # we always have relative sequence position. However, there is a
        # mask value used internally to mask inter-chain pairs
        False

    def get_mask_value(self) -> Tensor:
        """Get value used to mask this feature"""
        return torch.tensor([self.num_classes])

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        res_posns = batch["res_ids"]
        feat = rearrange(res_posns, "b n -> b n ()") - rearrange(
            res_posns, "b n -> b () n"
        )
        encoded_feat = feat.clone()
        if self.encoding_ty != FeatureEncodingTy.FOURIER:
            encoded_feat = bin_encode(feat, bins=torch.tensor(self.sep_bins))
        # Mask inter-chain pairs
        mask_val = self.get_mask_value().to(
            device=encoded_feat.device, dtype=feat.dtype
        )
        inter_chain_mask = rearrange(batch["chain_ids"], "b n -> b n ()") != rearrange(
            batch["chain_ids"], "b n -> b () n"
        )
        encoded_feat[inter_chain_mask] = mask_val
        feat[inter_chain_mask] = mask_val

        return map(lambda x: x.unsqueeze(-1), (encoded_feat, feat))


class RelativeDistance(FeatureGenerator):
    def __init__(
        self,
        one_hot: bool = False,
        rbf: bool = False,
        num_classes: int = 12,
        min_value: float = 2.5,
        max_value: float = 16.5,
        rbf_radii: List[float] = None,
        rbf_sigma: float = 4.0,
        atom_tys: List[str] = None,
    ):
        assert one_hot ^ rbf, f"one-hot or fourier encoding must be specified!"
        atom_tys = default(atom_tys, "CA CA N CA".split())
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.REL_DIST,
                encoding_ty=(
                    FeatureEncodingTy.ONEHOT if one_hot else FeatureEncodingTy.RBF
                ),
                num_classes=num_classes + 2,
                rbf_radii=default(rbf_radii, DEFAULT_PW_DIST_RADII),
                rbf_sigma=rbf_sigma,
                mult=len(atom_tys) // 2,
            )
        )
        self.bounds = (min_value, max_value)
        assert (
            len(atom_tys) % 2
        ) == 0, f"number of input atom types must be multiple of 2! got : {atom_tys}"
        self.atom_pairs = list(zip(atom_tys[::2], atom_tys[1::2]))
        self.rbf_radii = default(rbf_radii, DEFAULT_PW_DIST_RADII)
        self.rbf_sigma = rbf_sigma
        self.num_classes = num_classes

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_value(self) -> Tensor:
        """Get value used to mask this feature"""
        val = (
            (self.num_classes + 1)
            if self.encoding_ty == FeatureEncodingTy.ONEHOT
            else -1e5
        )
        return torch.tensor([val])

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        b, n = batch["coords"].shape[:2]
        min_dist, max_dist = self.bounds
        feat = torch.zeros(b, n, n, len(self.atom_pairs), device=batch["coords"].device)
        for idx, (a1, a2) in enumerate(self.atom_pairs):
            c1, c2 = map(
                lambda atom: batch["coords"][:, :, ATOM_TY_TO_ATOM_IDX[atom]], (a1, a2)
            )
            feat[..., idx] = torch.cdist(c1, c2)

        encoded_feat = feat.clone()
        if self.encoding_ty == FeatureEncodingTy.ONEHOT:
            normed_dists = (feat - min_dist) / (max_dist - min_dist)
            encoded_feat = 1 + torch.clamp(normed_dists, 0, 1) * self.num_classes
            encoded_feat[feat < min_dist] = 0

        return encoded_feat, feat


class RelativeChain(FeatureGenerator):
    def __init__(self):
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.REL_CHAIN,
                encoding_ty=FeatureEncodingTy.ONEHOT,
                num_classes=5,
                mult=1,
            )
        )

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return False

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """See super class"""
        (b, n), device = batch["coords"].shape[:2], batch["coords"].device
        chain_ids = default(batch["chain_ids"], torch.zeros(b, n, device=device))
        diffs = rearrange(chain_ids, "b i -> b i () ()") - rearrange(
            chain_ids, "b i -> b () i ()"
        )
        return 2 + torch.clamp(diffs, min=-2, max=2), chain_ids


class BindingInterface(FeatureGenerator):
    def __init__(
        self,
        contact_threshold: float = 10.0,
        atom_tys: List[str] = None,
        sample_fn=None,
        **kwargs,
    ):
        """
        Randomly samples a number of residues on the binding interface of a protein complex,
        and creates a one-hot vector indicating which feature was sampled.

        Parameters:
            sample_fn:
                Curried function to subsample the binding interface.
            contact_threshold:
                maximum pariwise distance under which two residues are considered to be in contact
            atom_tys:
                atom types to consider when computing pairwise distance (minimum over all atom types is taken)
        """
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.BINDING_INTERFACE,
                encoding_ty=FeatureEncodingTy.NONE,
                mult=1,
                embed_dim=1,
            )
        )
        atom_tys = default(atom_tys, ["CA", "CA"])
        self.atom_pairs = list(zip(atom_tys[::2], atom_tys[1::2]))
        self.contact_threshold = contact_threshold
        self.sample_fn = sample_fn

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_value(self) -> Tensor:
        return torch.tensor([0])

    def _generate(
        self,
        batch: Dict[str, Any],
        include_res_mask: Optional[Tensor] = None,
        precomputed_contacts: Optional[Tensor] = None,
    ) -> GenerateType:
        """
        Parameters:
            include_res_mask: boolean mask indicating which residues may be included
            as part of the binding interface
            Example: If we are generating binding interface features for an antibody-antigen complex,
            and we do not want to include paratope residues, then the include_res_mask would be TRUE
            for all residue positions corresponding to the antigen chain, and false elsewhere

            precomputed_contacts: boolean matrix indicating which resiudes form contacts
            pass this to skip recomputation, and allows for custom selection
        """
        b = batch["coords"].shape[0]
        if precomputed_contacts is not None:
            is_interface = precomputed_contacts.float()
        else:
            contacts = get_inter_chain_contacts(
                coords=batch["coords"],
                chain_ids=batch["chain_ids"],
                atom_pairs=self.atom_pairs,
                max_dist=self.contact_threshold,
            )
            # Take care of invalid residues
            valid_mask = batch["masks"]["valid"]["residue"]
            valid_contact_mask = torch.einsum(
                "... i, ... j -> ... i j", valid_mask, valid_mask
            ).bool()
            contacts[~valid_contact_mask] = False
            is_interface = torch.any(contacts, dim=-1).float()
            if include_res_mask is not None:
                is_interface[~include_res_mask] = 0

        encoded_feat = torch.zeros_like(is_interface)
        for i in range(b):
            interface_i = is_interface[i]
            if self.sample_fn is not None:
                interface_i = self.sample_fn(interface_i)
            encoded_feat[i] = interface_i

        return encoded_feat.unsqueeze(-1), is_interface.unsqueeze(-1)


class InterChainContacts(FeatureGenerator):
    def __init__(
        self,
        num_pairs_to_include: float = 0.33,
        contact_threshold: float = 10,
        atom_tys: List[str] = None,
    ):
        """
        Randomly samples a number of residue pairs in contact between chains in a protein complex.

        contacts are determined by contact_threshold param and contact_atom_tys param. Binary flags are
        generated and appended to pair features to indiacte contacting residues.

        Parameters:
            num_pairs_to_include: floating point value indicating the (maximum) number
                of residues pairs to indicate contacts for.
                If 0 < num_residues_to_include < 1, then the number of contacting pairs included
                will be treated as a geometric random variable with success probability
                `num_pairs_to_include` if this value is greater than 1, then the specified (integer)
                number of pairs will always be included
            max_dist:
                maximum pariwise distance under which two residues are considered to be in contact
            atom_tys:
                atom types to consider when computing pairwise distance (minimum over all atom types is taken)
        """
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.CONTACT,
                encoding_ty=FeatureEncodingTy.NONE,
                mult=1,
                embed_dim=1,
            )
        )
        atom_tys = default(atom_tys, ["CA", "CA"])
        self.atom_pairs = list(zip(atom_tys[::2], atom_tys[1::2]))
        self.contact_threshold = contact_threshold
        self.contact_atom_tys = atom_tys
        self._num_pairs_to_include = num_pairs_to_include

    def num_pairs_to_include(self, batch_size):
        if 0 < self._num_pairs_to_include < 1:
            seles = np.random.geometric(
                self._num_pairs_to_include, size=batch_size
            )  # - 1
            return torch.from_numpy(seles).to(torch.int32)
        return torch.tensor([int(self._num_pairs_to_include)] * batch_size).to(
            torch.int32
        )

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_value(self) -> Tensor:
        return torch.tensor([0])

    def _generate(
        self, batch: Dict[str, Any], include_res_mask: Optional[Tensor] = None
    ) -> GenerateType:
        """
        Parameters:
            include_res_mask: boolean mask indicating which residues may be included
                in contacting pairs. Note that after we apply this mask, we also apply
                the default structure_pair_mask from task.yaml. This option should only
                be specified if you want a more stringent exclusion criterion than the default structure_pair_mask.
                Example: If we are generating contacts for an antibody-antigen complex, we may wish
                to include only contacts between heavy chain cdrs and antigen.
        """
        b = batch["coords"].shape[0]
        contacts = get_inter_chain_contacts(
            coords=batch["coords"],
            chain_ids=batch["chain_ids"],
            atom_pairs=self.atom_pairs,
            max_dist=self.contact_threshold,
        )
        # Take care of invalid residues
        valid_mask = batch["masks"]["valid"]["residue"]
        valid_contact_mask = torch.einsum(
            "... i, ... j -> ... i j", valid_mask, valid_mask
        ).bool()
        if exists(include_res_mask):
            x = include_res_mask
            valid_contact_mask = valid_contact_mask & torch.einsum(
                "... i, ... j -> ... i j", x, x
            )
        contacts[~valid_contact_mask] = False

        num_to_include = self.num_pairs_to_include(b).to(contacts.device)
        num_to_include = torch.minimum(num_to_include, contacts.sum(dim=(-2, -1))).to(
            torch.int32
        )
        # Select specified number of random interface residue from interfaces
        contacts = contacts.float()
        encoded_feat = torch.zeros_like(contacts)

        for i in range(b):
            n_contacts = int(torch.sum(contacts[i]).item())
            if n_contacts == 0:
                continue
            contact_mask = contacts.new_zeros(n_contacts)
            contact_mask[: num_to_include[i]] = 1
            encoded_feat[i, contacts[i] > 0] = contact_mask[
                torch.randperm(len(contact_mask))
            ]
        return encoded_feat.unsqueeze(-1), contacts.unsqueeze(-1)


class InvariantRelativeOrientation(FeatureGenerator):
    def __init__(self):
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.REL_ORI,
                encoding_ty=FeatureEncodingTy.NONE,
            )
        )

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_value(self):
        """Apply mask to feature"""
        raise Exception("Not Yet Implemented")

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """see super class"""
        raise Exception("Not Yet Implemented")


class InvariantRelativeCoords(FeatureGenerator):
    def __init__(self):
        super().__init__(
            FeatureDescription(
                feature_id=FeatureID.REL_COORD,
                encoding_ty=FeatureEncodingTy.NONE,
            )
        )

    def can_mask(self):
        """Return whether or not this feature can be masked"""
        return True

    def get_mask_vaue(self):
        """Apply mask to feature"""
        raise Exception("Not Yet Implemented")

    def _generate(self, batch: Dict[str, Any]) -> GenerateType:
        """see super class"""
        raise Exception("Not Yet Implemented")
