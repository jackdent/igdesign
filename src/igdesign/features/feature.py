from __future__ import annotations
from typing import Tuple, Any, Optional, Dict

import torch
from torch import Tensor

from igdesign.utils import safe_to_device
from igdesign.features.feature_utils import default
from igdesign.features.feature_constants import (
    FeatureEncodingTy,
    FeatureID,
    FeatureTy,
    FeatureDescription,
)


class Feature:
    def __init__(
        self,
        description: FeatureDescription,
        encoded_data: Tensor,
        raw_data: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.description = description
        self.encoded_data = encoded_data.to(self.dtype)
        self.raw_data = raw_data
        self.kwargs = default(kwargs, {})  # store any extra information needed here

    @property
    def leading_shape(self) -> Tuple:
        return self.encoded_data.shape[:-1]

    @property
    def device(self):
        return self.encoded_data.device

    def to(self, device) -> Feature:
        data = (self.encoded_data, self.raw_data, self.kwargs)
        self.encoded_data, self.raw_data, self.kwargs = map(
            lambda x: safe_to_device(x, device), data
        )
        return self

    def pin_memory(self) -> Feature:
        self.encoded_data = self.encoded_data.pin_memory()
        return self

    def override(self, encoded_data: Tensor) -> Feature:
        curr_shape, new_shape = self.encoded_data.shape, encoded_data.shape
        assert (
            encoded_data.shape == self.encoded_data.shape
        ), f"shape mismatch {self}\n current: {curr_shape}\n new: {new_shape}"
        return Feature(
            description=self.description,
            encoded_data=encoded_data,
            raw_data=self.raw_data,
            kwargs=self.kwargs,
        ).to(self.device)

    @property
    def uid(self) -> str:
        return f"{self.name}:{self.feature_ty.value}"

    @property
    def name(self) -> str:
        return self.feature_id.value[0]

    @property
    def feature_id(self) -> FeatureID:
        return self.description.feature_id

    @property
    def feature_ty(self) -> FeatureTy:
        return self.description.feature_id.value[1]

    @property
    def encoding_ty(self) -> FeatureEncodingTy:
        return self.description.encoding_ty

    @property
    def dtype(self) -> torch.dtype:
        if self.encoding_ty in [FeatureEncodingTy.EMBED, FeatureEncodingTy.ONEHOT]:
            return torch.long
        return torch.get_default_dtype()

    def __repr__(self) -> str:
        return f"Feature : {self.name}"

    def __getattr__(self, attr):
        """Called only if this class does not have the given attribute"""
        try:
            return self.kwargs[attr]
        except:
            raise AttributeError(f"No attribute {attr} found for this class")
