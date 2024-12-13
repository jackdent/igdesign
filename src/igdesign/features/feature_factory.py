from __future__ import annotations
from enum import Enum
from typing import Dict, List, Union, Optional
from collections import OrderedDict, defaultdict

from hydra.utils import instantiate
import torch

from igdesign.utils import resolve_mask, resolve_pair_mask
from igdesign.features.feature import Feature
from igdesign.features.generator import FeatureGenerator
from igdesign.features.feature_utils import default, pair_mask_from_res_mask
from igdesign.features.feature_constants import (
    FeatureTy,
    FeatureID,
    FEATURE_TYS,
    FeatureDescription,
)

# Convenience type
FeatureKey = Union[str, Feature, FeatureDescription, FeatureGenerator, FeatureID]


class FeatureFactory:
    """
    Class for handling:
    - Feature Generation
    - Feature Masking
    """

    def __init__(
        self,
        generators: List[FeatureGenerator],
        masked_sequence_res_selection,
        masked_structure_res_selection,
        masked_structure_pair_selection,
    ):
        self.generators = OrderedDict()
        self.masked_sequence_res_selection = masked_sequence_res_selection
        self.masked_structure_res_selection = masked_structure_res_selection
        self.masked_structure_pair_selection = masked_structure_pair_selection

        for gen in generators:
            self.generators[gen.feature_name] = gen

    def __getitem__(self, item: Union[FeatureID, str]) -> FeatureGenerator:
        item = item.value[0] if isinstance(item, Enum) else item
        return self.generators[item]

    def __setitem__(self, key: FeatureKey, value: FeatureGenerator) -> None:
        key = self._get_name(key)
        assert key not in self.generators
        self.generators[key] = value

    def __contains__(self, item: FeatureKey) -> bool:
        return self._get_name(item) in self.generators

    def items(self):
        for g in self.generators:
            yield self._get_name(g), g

    def override(self, *generators) -> FeatureFactory:
        """
        Reset feature generators

        Example:

            #Override binding interface and contact features for inference:

            from linksar.features.generator import BindingInterface, InterChainContacts

            override_gens = [
                BindingInterface(num_residues_to_include=0),
                InterChainContacts(num_pairs_to_include=0),
            ]
            self.feature_factory.override(*override_gens)

        Returns:
           Feature factory with given generators overridden
        """
        for gen in generators:
            name = self._get_name(gen)
            if name in self:
                self[self._get_name(gen)] = gen
            else:
                print(
                    f"[WARNING] Feature generator {name}, does not exist! Can't override"
                )
        return self

    def feature_descriptions(
        self, feature_tys: Optional[Union[FeatureTy, List[FeatureTy]]] = None
    ) -> List[FeatureDescription]:
        feature_tys = default(feature_tys, FEATURE_TYS)
        feature_tys = (
            list(feature_tys) if not isinstance(feature_tys, list) else feature_tys
        )
        return [gen.description for gen in self.generators.values()]

    @property
    def feature_names(self) -> List[str]:
        return list(self.generators.keys())

    @property
    def feature_ids(self) -> List[FeatureID]:
        return [d.feature_id for d in self.feature_descriptions()]

    def _get_name(self, feature: FeatureKey):
        if isinstance(feature, FeatureID):
            feature = feature.value[0]
        elif not isinstance(feature, str):
            feature = (
                feature.name if isinstance(feature, Feature) else feature.feature_name
            )
        return feature

    def can_mask(self, feature: FeatureKey) -> bool:
        return self.generators[self._get_name(feature)].can_mask()

    def get_masks(self, batch, apply_valid_mask=True, apply_input_mask=True):
        """
        Get input masks for features. Default to applying both the valid residue mask and the input mask
        Args:
            batch (dict): batch of data
            apply_valid_mask (bool): whether to apply the valid residue mask
            apply_input_mask (bool): whether to apply the input mask
        Returns:
            dict: dictionary of masks
        """
        valid_res_mask = batch["masks"]["valid"]["residue"]
        N, L = valid_res_mask.shape
        sequence_res_mask = torch.zeros_like(valid_res_mask)
        structure_res_mask = torch.zeros_like(valid_res_mask)
        structure_pair_mask = torch.zeros(
            N, L, L, dtype=bool, device=valid_res_mask.device
        )

        if apply_valid_mask:
            valid_pair_mask = pair_mask_from_res_mask(~valid_res_mask)
            sequence_res_mask = ~valid_res_mask
            structure_res_mask = ~valid_res_mask
            structure_pair_mask = valid_pair_mask

        if apply_input_mask:
            if self.masked_sequence_res_selection:
                input_seq_mask = resolve_mask(
                    self.masked_sequence_res_selection, batch["masks"], subset="residue"
                )
                sequence_res_mask = sequence_res_mask | input_seq_mask
            if self.masked_structure_res_selection:
                input_struct_mask = resolve_mask(
                    self.masked_structure_res_selection,
                    batch["masks"],
                    subset="residue",
                )
                structure_res_mask = structure_res_mask | input_struct_mask
            if self.masked_structure_pair_selection:
                input_pmask = resolve_pair_mask(
                    self.masked_structure_pair_selection,
                    batch["masks"],
                    subset="residue",
                )
                structure_pair_mask = structure_pair_mask | input_pmask

        return {
            FeatureTy.SEQUENCE_RES: sequence_res_mask,
            FeatureTy.STRUCTURE_RES: structure_res_mask,
            FeatureTy.STRUCTURE_PAIR: structure_pair_mask,
        }

    def apply_masks(
        self,
        features: Dict[str, Feature],
        masks: Dict,
    ) -> Dict[str, Feature]:
        """Apply masks to features.
        NOTE: If `include_feats` is specified, then only these features will have masks applied
        Args:
            residue_mask mask to apply to residue features: _description_
            inter_chain_pair_mask (Optional[Tensor]): mask to apply to inter-chain pair features
            intra_chain_pair_mask (Optional[Tensor]): mask to apply to intra-chain pair features
            include_features (Optional[List[FeatureID]]): list of feature ids to include during masking
        """
        masked_feats = dict()
        for name, gen in self.generators.items():
            if name not in features:
                continue
            feature = features[name]
            if gen.can_mask():
                try:
                    feature = gen.apply_mask(feature, masks[feature.feature_ty])
                except KeyError:
                    raise ValueError(
                        "Feature type {feat.feature_ty} is not supported \
                        for masking. Either return False from gen.can_mask() or change \
                        the feature type. Note that sequence pair masks are not yet supported in configs, since we \
                        have not yet found any sequence pair features that require masking. If this \
                        ability is required, masked_sequence_pair_selection should be added to all feature configs"
                    )
            masked_feats[name] = feature
        return masked_feats

    def generate(
        self,
        batch: dict,
        apply_valid_mask=True,
        apply_input_mask=True,
        **override_features,
    ) -> Dict[str, Feature]:
        valid_res_mask = batch["masks"]["valid"]["residue"]
        assert valid_res_mask.dtype == torch.bool, f"mask dtype: {valid_res_mask.dtype}"
        gen_kwargs = batch.get("feature_kwargs", defaultdict(dict))
        feats = {
            name: gen.generate(batch=batch, **gen_kwargs.get(name, {}))
            for name, gen in self.generators.items()
        }
        if apply_valid_mask or apply_input_mask:
            masks = self.get_masks(
                batch,
                apply_valid_mask=apply_valid_mask,
                apply_input_mask=apply_input_mask,
            )
            feats = self.apply_masks(features=feats, masks=masks)

        override_features = override_features | batch.get("feature_override", {})
        for k, v in override_features.items():
            if k in feats:
                feats[k] = feats[k].override(v)

        return feats


def init_feature_factory(config, **kwargs):
    if config is None:
        return None
    generators = []
    for name, ind_conf in config.items():
        if ind_conf["use"]:
            target = {"_target_": ind_conf["_target_"]}
            gen_kwargs = (
                ind_conf["feature_args"] if "feature_args" in ind_conf else dict()
            )
            generators.append(instantiate(target, **gen_kwargs))
    return FeatureFactory(generators, **kwargs)
