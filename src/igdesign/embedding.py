from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import Tensor, nn, IntTensor
from torch.nn.functional import one_hot as to_one_hot
import pytorch_lightning as pl
from einops import repeat

from igdesign.features.feature_utils import (
    default,
    exists,
    fourier_encode,
)
from igdesign.features.feature_factory import FeatureFactory, FeatureKey
from igdesign.features.feature import Feature
from igdesign.features.feature_constants import (
    FeatureDescription,
    FeatureEncodingTy,
    FEATURE_NAMES,
    RES_FEATURE_TYS,
    PAIR_FEATURE_TYS,
)


def get_encoding_kwargs(feature_description: FeatureDescription):
    d, kwargs = feature_description, dict()
    if d.encoding_ty == FeatureEncodingTy.EMBED:
        kwargs = dict(num_classes=d.num_classes, embed_dim=d.embed_dim)
    if d.encoding_ty == FeatureEncodingTy.ONEHOT:
        kwargs = dict(num_classes=d.num_classes)
    if d.encoding_ty == FeatureEncodingTy.RBF:
        kwargs = dict(radii=d.rbf_radii, sigma=d.rbf_sigma)
    if d.encoding_ty == FeatureEncodingTy.FOURIER:
        kwargs = dict(n_feats=d.num_fourier_feats)
    kwargs.update(dict(mult=d.mult))
    return kwargs


class FeatureEmbedding(pl.LightningModule):
    def __init__(self, dim_out: int, mult: int):
        super().__init__()
        self.dim_out = dim_out
        self.mult = mult

    @property
    def embedding_dim(self) -> int:
        """Raw feature dimension"""
        return self.dim_out * self.mult if self.mult else self.dim_out

    def embed(self, feat: Feature) -> Tensor:
        embedded_feat = self.forward(feat)
        out_shape = (*feat.leading_shape, self.embedding_dim)
        return embedded_feat.reshape(out_shape)


class Embedding(FeatureEmbedding):
    """nn.Embedding (wrapped to standardize)"""

    def __init__(self, num_classes: int, embed_dim: int, mult: int):
        super().__init__(dim_out=embed_dim, mult=mult)
        self.offsets = None
        self.num_classes = num_classes
        self.embedding = nn.Embedding(mult * num_classes, embed_dim)

    def get_offsets(self, feat: Tensor):
        """Values to shift bins by for case of multiple embeddings"""
        if not exists(self.offsets):
            offsets = [i * self.num_classes for i in range(self.mult)]
            self.offsets = torch.tensor(offsets, device=feat.device, dtype=torch.long)
        return self.offsets

    def forward(self, feat: Feature):
        """Embed the feature"""
        to_emb = feat.encoded_data
        assert to_emb.shape[-1] == self.mult, f"{to_emb.shape},{self.mult},{feat.name}"
        max_token = torch.max(to_emb + self.get_offsets(to_emb))
        min_token = torch.min(to_emb + self.get_offsets(to_emb))
        expected_max_token = self.mult * self.num_classes
        assert min_token >= 0, f"[Embedding] got negative token value {min_token}"
        assert max_token < expected_max_token, (
            f"[Embedding] got value outside of embedding range, [{feat.name}]: max token "
            f"expected: {expected_max_token}, max token actual: {max_token}"
        )
        return self.embedding(to_emb + self.get_offsets(to_emb))


class OneHot(FeatureEmbedding):
    """One Hot encoding (wrapped, so it can be used in module dict)"""

    def __init__(self, num_classes, mult: int = 1, std_noise: float = 0):
        """One hot feature encoding

        :param num_classes: number of classes to encode
        :param mult: trailing dimension of input features
        :param std_noise: [Optional] noise standard deviation (added to encodings)
        """
        super().__init__(dim_out=num_classes, mult=mult)
        self.std_noise = std_noise

    def forward(self, feat: Feature):
        """One hot encode the features"""
        to_hot = feat.encoded_data
        assert to_hot.shape[-1] == self.mult, f"{to_hot.shape},{self.mult},{feat.name}"
        assert torch.min(to_hot) >= 0, f"{feat.name}:{torch.min(to_hot)},{self.dim_out}"
        assert (
            torch.max(to_hot) < self.dim_out
        ), f"{feat.name}:{torch.max(to_hot)},{self.dim_out}"
        encoding = to_one_hot(to_hot, self.dim_out)
        encoding = encoding + torch.randn_like(encoding.float()) * self.std_noise
        return encoding.detach()


class RBF(FeatureEmbedding):
    """RBF Encoding (wrapped, so it can be used in module dict)"""

    def __init__(
        self,
        radii: List[float],
        mult: int = 1,
        sigma: Optional[float] = None,
        exp_clamp: float = 16,
        std_noise: float = 0,
    ):
        """RBF feature encoding"""
        super().__init__(
            dim_out=len(radii),
            mult=mult,
        )
        radii = torch.tensor(radii).float()
        self.radii = repeat(radii, "r -> m r", m=mult)
        self.sigma_sq = default(sigma, torch.mean(radii[1:] - radii[:-1])) ** 2
        self.exp_clamp = exp_clamp
        self.std_noise = std_noise

    def forward(self, feat: Feature):
        """RBF - encode the features"""
        raw_data = feat.encoded_data
        assert (
            raw_data.shape[-1] == self.mult
        ), f"{raw_data.shape},{self.mult},{feat.name}"
        raw_data = raw_data.unsqueeze(-1)
        shape_diff = raw_data.ndim - self.radii.ndim
        radii = self.radii[(None,) * shape_diff].to(raw_data.device)
        exp_val = torch.square((radii - raw_data) / self.sigma_sq)
        clamped_exp_val = torch.clamp_max(exp_val, self.exp_clamp)
        encoding = torch.exp(-clamped_exp_val)
        encoding[clamped_exp_val == self.exp_clamp] = 0
        encoding = encoding + torch.randn_like(encoding.float()) * self.std_noise
        return encoding.detach()


class Fourier(FeatureEmbedding):  # noqa
    """Fourier (sin and cos) encoding (wrapped so it can be used in module dict)"""

    def __init__(self, n_feats, include_self=False, mult: int = 1):  # noqa
        super().__init__(
            dim_out=2 * n_feats + int(include_self),
            mult=mult,
        )
        self.n_feats, self.include_self = n_feats, include_self

    def forward(self, feat: Feature):
        """Fourier encode the features"""
        with torch.no_grad():
            to_encode = feat.encoded_data
            assert (
                to_encode.shape[-1] == self.mult
            ), f"{to_encode.shape},{self.mult},{feat.name}"
            return fourier_encode(
                to_encode, num_encodings=self.n_feats, include_self=self.include_self
            )


class InputEmbedding(pl.LightningModule):  # noqa
    """Input Embedding"""

    def __init__(
        self,
        factory: FeatureFactory,
        res_embed_dim: Optional[int] = None,
        pair_embed_dim: Optional[int] = None,
    ):
        super(InputEmbedding, self).__init__()
        self.factory = factory
        feature_descriptions = factory.feature_descriptions()
        self.residue_embeddings, self.pair_embeddings = self._init_embeddings(
            feature_descriptions
        )
        self.res_dim = self._count_embedding_dim(self.residue_embeddings)
        self.pair_dim = self._count_embedding_dim(self.pair_embeddings)

        # Optional projections mapping to hidden dimension
        self.res_project_in = (
            nn.Sequential(
                nn.Linear(self.res_dim, res_embed_dim), nn.LayerNorm(res_embed_dim)
            )
            if exists(res_embed_dim)
            else nn.Identity()
        )

        self.pair_project_in = (
            nn.Sequential(
                nn.Linear(self.pair_dim, pair_embed_dim), nn.LayerNorm(pair_embed_dim)
            )
            if exists(pair_embed_dim)
            else nn.Identity()
        )

    def forward(
        self,
        features: Optional[Dict[str, Feature]] = None,
        args: Optional[Dict] = None,
        return_as_dict: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
        """Get pair and residue input features"""
        assert exists(features) ^ exists(args)
        features = self.factory.generate(args) if not exists(features) else features
        residue_feats = self._get_input(features, RES_FEATURE_TYS, return_as_dict)
        pair_feats = self._get_input(features, PAIR_FEATURE_TYS, return_as_dict)
        if return_as_dict:
            return residue_feats, pair_feats
        return self.res_project_in(residue_feats), self.pair_project_in(pair_feats)

    @property
    def feature_dims(self) -> Tuple[int, int]:
        """Residue and Pair Feature dimension"""
        return self.res_dim, self.pair_dim

    @staticmethod
    def _count_embedding_dim(embeddings) -> int:
        """sums output dimension of list of embeddings"""
        return sum([e.embedding_dim for e in embeddings.values()])

    def __contains__(self, key: FeatureKey) -> bool:
        return key in self.factory

    def get_feature_embedding_dim(self, key: FeatureKey) -> int:
        name = self.factory._get_name(key)
        if name in self.residue_embeddings:
            return self.residue_embeddings[name].embedding_dim
        elif name in self.pair_embeddings:
            return self.pair_embeddings[name].embedding_dim
        else:
            err_msg = (
                f"no embedding found for feature: {name}\n"
                f"residue features: {[k for k in self.residue_embeddings]}\n"
                f"pair features: {[k for k in self.pair_embeddings]}"
            )
            raise Exception(err_msg)

    @staticmethod
    def _init_embeddings(
        feature_descriptions: List[FeatureDescription],
    ) -> Tuple[nn.ModuleDict, nn.ModuleDict]:
        """
        Gets residue and pair input dimensions as well as embedding/encoding
        functions for each input feature.

        Feature types can be found in features/input_features.py
        """
        res_embeddings, pair_embeddings = nn.ModuleDict(), nn.ModuleDict()
        for d in feature_descriptions:
            if d.feature_ty in RES_FEATURE_TYS:
                embed_dict = res_embeddings
            else:
                embed_dict = pair_embeddings
            kwargs = get_encoding_kwargs(d)
            embed_dict[d.feature_name] = EMBED_TY_TO_EMBED_CLASS[d.encoding_ty](
                **kwargs
            )
        return res_embeddings, pair_embeddings

    def _get_input(
        self,
        features: Dict[str, Feature],
        feature_tys: list,
        return_feature_dict: bool = False,
    ) -> Optional[Union[Dict[str, Tensor], Tensor]]:
        """Get residue input features"""
        embedded_feats = {} if return_feature_dict else []
        embed_dict = (
            self.residue_embeddings
            if feature_tys == RES_FEATURE_TYS
            else self.pair_embeddings
        )
        # Iterate over names so concatenation order is always consistent
        for feat_name in FEATURE_NAMES:
            if feat_name not in features:
                continue
            feat = features[feat_name].to(self.device)
            if feat.feature_ty in feature_tys:
                if feat.name not in embed_dict:
                    raise Exception(f"No Embedding found for feature: {feat}")
                embedded_feat = embed_dict[feat.name].embed(feat)
                if torch.any(torch.isnan(embedded_feat)):
                    raise ValueError(f"Embedded feature {feat.name} contains nan!")
                if return_feature_dict:
                    embedded_feats[feat_name] = embedded_feat
                else:
                    embedded_feats.append(embedded_feat)
        if return_feature_dict:
            return embedded_feats
        return torch.cat(embedded_feats, dim=-1) if len(embedded_feats) > 0 else None


EMBED_TY_TO_EMBED_CLASS = {
    FeatureEncodingTy.EMBED: Embedding,
    FeatureEncodingTy.FOURIER: Fourier,
    FeatureEncodingTy.ONEHOT: OneHot,
    FeatureEncodingTy.RBF: RBF,
}
