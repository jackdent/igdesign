import yaml
from pathlib import Path
from copy import deepcopy
from typing import MutableSequence, MutableMapping

import torch
import hydra
import numpy as np
import cytoolz as ct
import itertools as it

import igdesign.structure_utils as su
from igdesign.structure_utils import nan_to_ca
from igdesign.tokenization import get_chi_mask_batch, get_chi_symm_batch, tokenize
from igdesign.data.pdb_parser import PDBParserAnnotator
from igdesign.data.structure_analysis import annotate_chain
from igdesign.data.datasets.structural import (
    StructureDataset,
    apply_crop_mask,
    get_idx_mask,
    get_valid_mask,
)


def load_yaml(path):
    with open(path) as handle:
        item = yaml.safe_load(handle)
    return item


def imgt_iterator():
    """
    Returns an enumerated iterator over IMGT regions as tuples (i, (chain, region)).
    Use this to get consistent region numbering in different places in the code.
    """
    chains = ["heavy", "light"]
    regions = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]
    return enumerate(it.product(chains, regions))


class PdbAntibodyDataset(StructureDataset):
    def __init__(
        self,
        configs=[],
        include_sidechains=False,
        include_heavy_chains=True,
        include_light_chains=True,
        include_antigens=True,
        ag_crop_size=200,
        ag_crop_method="closest_continuos",
        ag_crop_reference="epitope",
        ag_noise=0,
        ab_noise=0,
        keep_raw={"antigen"},
    ):
        """
        Initialize the PdbAntibodyDataset.

        Parameters
        ----------
        configs : List[Dict]
            A list of configuration dictionaries for each PDB structure
            or a list of yaml files or a list of folders with the yaml files
        include_sidechains : bool, default=False
            A boolean indicating whether to include side chain coordinates in the output, by default False.
        """
        super(PdbAntibodyDataset, self).__init__(
            keep_raw=keep_raw, include_sidechains=include_sidechains
        )

        self.configs = list(
            ct.concat(self.maybe_load(item) for item in self.maybe_load(configs))
        )
        self.configs = [
            ct.assoc(deepcopy(c), "sample_idx", i)
            for c in self.configs
            for i in range(c["num_samples"])
        ]
        for config in self.configs:
            config["name"] = config["name"] + f"_{config['sample_idx']}"
        self.include_heavy_chains = include_heavy_chains
        self.include_light_chains = include_light_chains
        self.include_antigens = include_antigens
        self.ag_crop_size = ag_crop_size
        self.ag_crop_method = ag_crop_method
        self.ag_crop_reference = ag_crop_reference
        self.ag_noise = ag_noise
        self.ab_noise = ab_noise
        self.pad_token_dict["feature_override"] = dict(binding_interface=0, contact=0)
        self.mask_keys.extend(["antibody", "antigen", "framework"])
        self.mask_keys.extend(
            [chain for chain in ("heavy", "light") if self.use(chain)]
        )
        self.mask_keys.extend(
            [chain[0] + key for _, (chain, key) in imgt_iterator() if self.use(chain)]
        )

    def maybe_load(self, item):
        if isinstance(item, str) or isinstance(item, Path):
            if Path(item).is_file():
                item = load_yaml(item)
            elif Path(item).is_dir():
                item = [load_yaml(file) for file in Path(item).glob("*.yaml")]
            else:
                raise Exception(
                    f"Configuration file {item} is neither a valid file or a valid dict!"
                )
        return item if isinstance(item, MutableSequence) else [item]

    def __getitem__(self, idx):
        return self.get_item(self.configs[idx])

    def __len__(self):
        return len(self.configs)

    def use(self, chain):
        if chain == "antigens":
            return self.include_antigens and self.ag_crop_size > 0
        elif chain == "heavy":
            return self.include_heavy_chains
        elif chain == "light":
            return self.include_light_chains
        else:
            raise ValueError(f"I don't know if I should use {chain}!")

    def get_item(self, cfg):
        """Return a dictionary of PDB structure information for a given configuration.

        Parameters
        ----------
        cfg : dict
            The configuration dictionary for the PDB structure.

        Returns
        -------
        dict
            A dictionary of PDB structure information for the given configuration.

        Example
        -------
        cfg = {
            'pdb': '7sue',
            'pdb_path': './pdbs/7sue.pdb',
            'heavy': {'chain': 'F', 'has_sequence': True, 'has_coords': True, "sequence": None},
            'light':{'chain': 'E', 'has_sequence': True, 'has_coords': True, 'coords': None},
            "antigens": [{"chain": 'J', "has_sequence": True, "has_coords": True, "sequence": None}],
            "epitopes": {'J': [36, 37, 80, 81, 82, 83, 84, 86, 90, 91, 92, 93, 94, 95, 96, 97]}
            }
        dataset = PdbAntibodyDataset()
        item = dataset.get_item(cfg)
        """
        cfg = deepcopy(cfg)
        item = self.item_from_cfg(cfg)

        chain_symbols, sequences = ct.get(["chains", "sequence"], item)
        coords = torch.concat(item["coords"])
        chain_ids = torch.concat(
            [torch.full([len(s)], i) for i, s in enumerate(sequences)]
        )
        res_ids = torch.concat([torch.arange(len(s)) for s in sequences])

        res_mask, atom_mask = ct.get(["residue", "atom"], get_valid_mask(coords))
        chi_mask, chi_symm = self.get_chi_masks(sequences)
        record = {
            "tokenized_sequences": tokenize("".join(sequences)),
            "coords": coords,
            "chain_ids": chain_ids,
            "res_ids": res_ids,
            "res_mask": res_mask,
            "atom_mask": atom_mask,
            "chi_mask": chi_mask,
            "chi_symm": chi_symm,
        }
        annotations = self.get_annotations(item)
        masks = self.get_masks(item, chain_ids, chain_symbols, annotations, res_mask)

        raw = self.save_raw_antigen(item)
        crop_mask = self.get_antigen_crop_mask(coords, masks, res_mask)
        crop_mask |= masks["antibody"]
        record = apply_crop_mask(record, crop_mask)
        masks = apply_crop_mask(masks, crop_mask)
        record = self.finalize_masks(record, masks)
        record["chain_symbols"] = chain_symbols
        for key in ["pdb", "name"]:
            record[key] = item[key]

        record["feature_override"] = self.add_feature_overrides(cfg, record)
        expected_centroid = cfg["heavy"].get("expected_centroid")
        if expected_centroid is not None:
            record["coord_mean"] = torch.tensor(expected_centroid, dtype=torch.float)
        record["coords"] = torch.nan_to_num_(
            nan_to_ca(record["coords"], record["masks"]["valid"]["atom"]), 0
        )
        record.update(raw)
        record["sequences"] = sequences
        return record

    def save_raw_antigen(self, item):
        sequences, coords = {}, {}
        for chain in item.get("antigens", []):
            idx = item["chains"].index(chain)
            sequences[chain] = item["sequence"][idx]
            coords[chain] = item["coords"][idx]
        return {"raw_sequences": sequences, "raw_coords": coords}

    def get_antigen_crop_mask(self, coords, masks, res_mask):
        antigen_mask = masks["antigen"]
        if self.ag_crop_method == "none":
            return antigen_mask.clone()
        if self.ag_crop_method == "custom":  # exact residues to be kept are specifed
            all_idx = torch.arange(len(res_mask))
            antigen_idx = all_idx[antigen_mask]
            selected_idx = antigen_idx[
                torch.tensor(self.ag_crop_reference, dtype=torch.long)
            ]

            crop_mask = torch.zeros_like(res_mask)
            crop_mask[selected_idx] = True
        else:
            crop_ref_mask = masks[self.ag_crop_reference]
            crop_mask = su.crop(
                coords,
                antigen_mask,
                crop_ref_mask,
                res_mask,
                self.ag_crop_size,
                self.ag_crop_method,
            )
        return crop_mask

    def item_from_cfg(self, cfg):
        item = dict(
            pdb=cfg.get("pdb", ""), chains=[], sequence=[], coords=[], annotations=[]
        )
        for key in ["pdb", "name", "num_samples", "sample_idx"]:
            item[key] = cfg.get(key, "")
        for key in ("heavy", "light"):
            if self.use(key) and key in cfg:
                item = self.add_immunoglobulin(cfg, key, item)
        if self.use("antigens") and "antigens" in cfg:
            item = self.add_antigens(cfg, item)
        item = self.to_torch(item)
        return item

    def parse_chain(self, data, pdb_path=None):
        if isinstance(pdb_path, MutableSequence):
            pdb_path_idx = data.get("pdb_file_index", 0)
            pdb_path = pdb_path[pdb_path_idx]
        chain = data["chain"]
        should_parse = False
        if data["has_sequence"] and data.get("sequence") is None:
            should_parse = True
        if data["has_coords"] and data.get("coords") is None:
            should_parse = True
        if should_parse:
            parser = PDBParserAnnotator(verbal_flag=False)
            parsed = parser.parse_PDB(pdb_path, chain)
        sequence = ""
        if data["has_sequence"]:
            sequence = data.get("sequence") or parsed["sequence"]
        natoms = 37 if self.include_sidechains else 4
        if data["has_coords"]:
            coords = data.get("coords") or (parsed["coords"][:, :natoms, :])
        else:
            coords = np.random.randn(len(sequence), natoms, 3)
            if "expected_centroid" in data:
                centroid = np.array(data["expected_centroid"])[
                    np.newaxis, np.newaxis, ...
                ]
                coords += centroid
        return chain, sequence, coords

    def add_immunoglobulin(self, cfg, key, item):
        # this function adds an antibody chain to the batch.
        # it assumes a single Fv domain per chain and will not work for ScFv chains
        # for indexing safety, we add expectation that the chain is trimmed to Fv region
        
        data, pdb_path = cfg.get(key), cfg.get("pdb_path")
        if "regions" in data:
            data = self.make_ig_from_regions(data)
        chain, sequence, coords = self.parse_chain(data, pdb_path)
        annotations = data.get("annotations", {})

        if data["has_sequence"]:
            if not annotations:
                imgt = annotate_chain(sequence)["aa_regions"]
                annotations = {k.split("_")[1]: v for k, v in imgt.items()}
            start = annotations["fwr1"][0]
            end = annotations["fwr4"][1]

            if start > 0 or end < len(sequence):
                raise ValueError(f"Trim antibody chains to Fv region only prior to passing to IGDesign"
                                 f"For chain {key}, Fv start: {start}, Fv end: {end}, len(sequence): {len(sequence)}")
            
            sequence = sequence[start:end]
            coords = coords[start:end]

            # if start > 0:
            #     annotations = {
            #         k: (v[0] - start, v[1] - start) for k, v in annotations.items()
            #     }
            
        item[key] = chain
        item["chains"].append(chain)
        item["sequence"].append(sequence)
        item["coords"].append(coords + np.random.randn(*coords.shape) * self.ab_noise)
        item["annotations"].append(annotations)
        return item

    def make_ig_from_regions(self, config):
        region_cfg = config["regions"]
        fwr = [f"fwr{i}_seq" for i in range(1, 5)]
        fwr_seq = ct.get(fwr, region_cfg)
        fwr_len = [len(x) for x in fwr_seq]

        cdr = [f"cdr{i}_len" for i in range(1, 4)]
        cdr_len = ct.get(cdr, region_cfg)
        cdr_seq = ["G" * x for x in cdr_len]

        sequence = "".join(ct.interleave((fwr_seq, cdr_seq)))
        csum = ct.reduce(
            lambda acc, x: acc + [acc[-1] + x], ct.interleave((fwr_len, cdr_len)), [0]
        )
        annotations = {
            k.split("_")[0]: v
            for k, v in dict(
                zip(ct.interleave([fwr, cdr]), ct.sliding_window(2, csum))
            ).items()
        }
        config["sequence"] = sequence
        config["annotations"] = annotations
        return config

    def add_antigens(self, cfg, item):
        pdb_path = cfg.get("pdb_path")
        chains, sequences, coords = zip(
            *(self.parse_chain(d, pdb_path) for d in cfg["antigens"])
        )
        coords = [c + np.random.randn(*c.shape) * self.ab_noise for c in coords]
        annotations = {
            "epitope": self._get_epitope_idx(cfg, c, s)
            for c, s in zip(chains, sequences)
        }

        item["antigens"] = chains
        item["chains"].extend(chains)
        item["sequence"].extend(sequences)
        item["coords"].extend(coords)
        item["annotations"].append(annotations)
        return item

    def get_chi_masks(self, sequences):
        tokens = [tokenize(s) for s in sequences]
        chi_mask = [
            get_chi_mask_batch(t.unsqueeze(0)).squeeze(0).bool() for t in tokens
        ]
        chi_symm = [
            get_chi_symm_batch(t.unsqueeze(0)).squeeze(0).bool() for t in tokens
        ]
        return torch.cat(chi_mask), torch.cat(chi_symm)

    def get_annotations(self, item):
        annotations = {}
        for idx, chain in enumerate(item["chains"]):
            for key, value in item["annotations"][idx].items():
                if chain == item.get("heavy") or chain == item.get("light"):
                    let = "h" if chain == item["heavy"] else "l"
                    annotations[let + key] = {idx: list(range(*value))}
                else:
                    if key in annotations:
                        annotations[key][idx] = value
                    else:
                        annotations[key] = {idx: value}
        return annotations

    def _get_epitope_idx(self, cfg, chain, chain_sequence):
        epitope = cfg["epitopes"].get(chain)
        if epitope is None:
            return None
        epitope_idx = epitope["indices"]
        if "residues" in epitope:
            epitope_seq = [chain_sequence[i] for i in epitope_idx]
            if any(x != y for x, y in zip(epitope_seq, epitope["residues"])):
                msg = "Parsed eptope sequence does not match the expected eptope sequence.\n"
                msg += f"Chain: {chain}, parsed: {''.join(epitope_seq)}, expected: {''.join(epitope['residues'])}"
                raise Exception(msg)
        return np.array(epitope_idx)

    def get_masks(self, item, chain_ids, chain_symbols, annotations, res_mask):
        masks = dict(all=res_mask)
        for key in ("heavy", "light"):
            if (key in item) and self.use(key):
                masks[key] = chain_ids == chain_symbols.index(item[key])
            else:
                masks[key] = torch.zeros_like(res_mask)
        masks["antibody"] = masks["heavy"] | masks["light"]
        masks["antigen"] = torch.zeros_like(res_mask)
        if self.use("antigens"):
            for chain in item["antigens"]:
                masks["antigen"] |= chain_ids == chain_symbols.index(chain)
        for key, value in annotations.items():
            masks[key] = ct.reduce(
                torch.logical_or,
                (get_idx_mask(i, idxs, chain_ids) for i, idxs in value.items()),
            )
        masks["framework"] = ct.reduce(
            torch.logical_or, [masks[k] for k in masks if "fwr" in k]
        )
        return masks

    def finalize_masks(self, record, masks):
        res_mask, atom_mask = record.pop("res_mask"), record.pop("atom_mask")
        masks = {k: {"residue": v} for k, v in masks.items()}
        for key, val in masks.items():
            mask = atom_mask.detach().clone()
            mask[~val["residue"]] = 0
            masks[key]["atom"] = mask
        masks["valid"] = {"residue": res_mask, "atom": atom_mask}
        record["masks"] = masks
        return record

    def add_feature_overrides(self, cfg, record):
        interface_mask = self.override_binding_interface(record, cfg)
        contact_mask = self.override_contact(record)
        return {"binding_interface": interface_mask, "contact": contact_mask}

    def override_binding_interface(self, record, cfg):
        subsample_fn = cfg.get("epitope_subsample")
        if isinstance(subsample_fn, MutableMapping):
            subsample_fn = hydra.utils.call(subsample_fn)
        masks, res_mask = record["masks"], record["masks"]["valid"]["residue"]
        epitope_mask = masks["epitope"]["residue"] & res_mask
        if subsample_fn is not None:
            epitope_mask = subsample_fn(epitope_mask)
        if cfg.get("cdrs_as_paratope", {}).get("use", False):
            paratope_mask = (
                ct.reduce(
                    torch.logical_or, [masks[k]["residue"] for k in masks if "cdr" in k]
                )
                & res_mask
            )
            subsample_fn = cfg.get("cdrs_as_paratope").get("subsample")
            if isinstance(subsample_fn, MutableMapping):
                subsample_fn = hydra.utils.call(subsample_fn)
            if subsample_fn is not None:
                paratope_mask = subsample_fn(paratope_mask)
            epitope_mask = epitope_mask & paratope_mask
        return epitope_mask.float().unsqueeze(dim=-1)

    def override_contact(self, record):
        length = record["masks"]["valid"]["residue"].shape[-1]
        contact_mask = torch.zeros(length, length, 1, dtype=torch.float32)
        return contact_mask
