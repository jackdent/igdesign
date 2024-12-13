from __future__ import annotations
from enum import Enum
from typing import NamedTuple, List, Optional

import numpy as np

from igdesign.tokenization import AA_TO_IDX, IDX_TO_AA
from igdesign.tokenization import ATOM_LIST


AA_ALPHABET = [IDX_TO_AA[i] for i in range(len(AA_TO_IDX))]
NATURAL_AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
NATURAL_AA_INDICES = [AA_TO_IDX[aa] for aa in NATURAL_AA_ALPHABET]
ATOM_TY_TO_ATOM_IDX = {a: ATOM_LIST.index(a) for a in ATOM_LIST}


# DEFAULTS
rng = np.arange(1, 33).tolist()
DEFAULT_SEP_BINS = [-i for i in reversed(rng)] + [0] + rng  # noqa
SMALL_SEP_BINS = [1e6, 30, 20, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
SMALL_SEP_BINS = [-i for i in SMALL_SEP_BINS] + [0] + list(reversed(SMALL_SEP_BINS))
DEFAULT_PW_DIST_RADII = np.linspace(0, 20, 16).tolist() + [25, 30, 40, 50]  # noqa
DEFAULT_CENTRALITY_RADII = [6, 12, 18, 24, 30, 36, 42, 48]
DEFAULT_ATOM_TY = "CA"
DEFAULT_COORD_IDX = ATOM_LIST.index(DEFAULT_ATOM_TY)


# ENUMS
class FeatureTy(Enum):
    """Feature Type flag"""

    (
        SEQUENCE_RES,
        SEQUENCE_PAIR,
        STRUCTURE_RES,
        STRUCTURE_PAIR,
        OTHER_RES,
        OTHER_PAIR,
    ) = (
        "sequence_res",
        "sequence_pair",
        "structure_res",
        "structure_pair",
        "other_res",
        "other_pair",
    )


class FeatureID(Enum):
    """Identifiers for each feature type.
    ORDER IS IMPORTANT WHEN EMBEDDING FEATURES,
    ONLY APPEND AND NEVER REORDER.
    """

    # Residue features
    RES_TY = (
        "res_ty",
        FeatureTy.SEQUENCE_RES,
    )  # RES_TY FEATURE MUST BE FIRST! DO NOT MOVE THIS!
    BB_DIHEDRAL = ("bb_dihedral", FeatureTy.STRUCTURE_RES)
    CENTRALITY = ("centrality", FeatureTy.STRUCTURE_RES)
    BINDING_INTERFACE = ("binding_interface", FeatureTy.STRUCTURE_RES)
    POS_ENC = ("pos_enc", FeatureTy.SEQUENCE_RES)
    TIMESTEP_ENC = ("timestep_enc", FeatureTy.OTHER_RES)
    TASK_ENC = ("task_enc", FeatureTy.OTHER_RES)

    # Pair features
    REL_SEP = ("rel_sep", FeatureTy.SEQUENCE_PAIR)
    REL_DIST = ("rel_dist", FeatureTy.STRUCTURE_PAIR)
    REL_ORI = ("rel_ori", FeatureTy.STRUCTURE_PAIR)
    TR_ORI = ("tr_ori", FeatureTy.STRUCTURE_PAIR)
    REL_COORD = ("rel_coord", FeatureTy.STRUCTURE_PAIR)
    REL_CHAIN = ("rel_chain", FeatureTy.SEQUENCE_PAIR)
    CONTACT = ("contact", FeatureTy.STRUCTURE_PAIR)

    # Placeholders for undefined features
    EXTRA_RES = ("extra_res", FeatureTy.OTHER_RES)  # Placeholder for undefined features
    EXTRA_PAIR = (
        "extra_pair",
        FeatureTy.OTHER_PAIR,
    )  # Placeholder for undefined features


FEATURE_NAMES = [e.value[0] for e in FeatureID]
FEATURE_IDS = [e for e in FeatureID]
FEATURE_NAME_TO_FEATURE_ID = {e.value[0]: e for e in FeatureID}
FEATURE_TYS = [e for e in FeatureTy]
RES_FEATURE_TYS = [FeatureTy.SEQUENCE_RES, FeatureTy.STRUCTURE_RES, FeatureTy.OTHER_RES]
PAIR_FEATURE_TYS = [
    FeatureTy.SEQUENCE_PAIR,
    FeatureTy.STRUCTURE_PAIR,
    FeatureTy.OTHER_PAIR,
]


class FeatureEncodingTy(Enum):
    """Feature Type flag"""

    RBF = "rbf"  # Radial basis encoding
    ONEHOT = "one-hot"  # One-Hot encoding
    EMBED = "embedding"  # nn.Embedding
    FOURIER = "fourier"  # Fourier Encoding (sin and cos)


ENCODING_TYS = [e for e in FeatureEncodingTy]
ENCODING_NAMES = [e.value for e in FeatureEncodingTy]
ENCODING_NAME_TO_ENCODING_TY = {e.value: e for e in FeatureEncodingTy}


class FeatureDescription(NamedTuple):
    """
    num_classes:
        number of classes to encode to (e.g. bins for one-hot).
        NOTE: Used only if FeatureEncodingTy is ONE_HOT or EMBED.
    num_fourier_feats:
        number of fourier features to use.
        NOTE: Used only if FeatureEncodingTy is FOURIER
    rbf_radii:
        Radii to use for rbf encoding
        NOTE: Used only if FeatureEncodingTy is RBF
    rbf_sigma:
        Sigma(s) to use for rbf encoding (defaults to avg. diff in radii)
        NOTE: Used only if FeatureEncodingTy is RBF
    embed_dim:
        Dimension of feature embedding.
        NOTE: Only relevant if FeatureEncodingTy is EMBED
    mult:
        Feature multiplicity - e.g. for bb-dihedral features, we encode phi, psi, and omega.
        the feature multiplicity here is 3. Similarly, for pairwise distance features, the number of
        different atom pairs constitutes the feature multiplicity.
    """

    feature_id: FeatureID
    encoding_ty: FeatureEncodingTy
    num_classes: Optional[int] = None
    num_fourier_feats: Optional[int] = None
    rbf_radii: Optional[List] = None
    rbf_sigma: Optional[float] = None
    embed_dim: Optional[int] = None
    mult: int = 1

    pretrained_model_name_or_path: Optional[str] = None
    use_pretrained_weights: bool = True
    freeze_lm: bool = True
    use_cache: bool = False
    cache_dir: Optional[str] = None
    cache_clear_dir: bool = False
    cache_verbose: bool = False

    @property
    def feature_name(self) -> str:
        return self.feature_id.value[0]

    @property
    def feature_ty(self) -> FeatureTy:
        return self.feature_id.value[1]

    def __repr__(self) -> str:
        uid = f"{self.feature_id.value[0]}-{self.feature_id.value[1].value}"
        return f"FeatureDescription [{uid}:{self.encoding_ty.value}]"
