import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from igdesign.tokenization import (
    AA_TO_IDX,
    get_chi_mask_batch,
    get_chi_symm_batch,
    tokenize,
)
from igdesign.structure_utils import nan_to_ca


def unpack_coordinates(compressed_coords, sequence):
    length = len(sequence)
    compressed_idxs = _coord_idxs(sequence)
    array = np.full(shape=(37 * length, 3), fill_value=np.nan, dtype=np.float32)
    array[compressed_idxs] = compressed_coords
    array = array.reshape(-1, 37, 3)
    return array


class StructureDataset:
    def __init__(self, include_sidechains=False, keep_raw=None, *args, **kwargs):
        super(StructureDataset, self).__init__(*args, **kwargs)
        self.keep_raw = keep_raw if keep_raw is not None else []
        self.include_sidechains = include_sidechains
        self.mask_keys = ["all", "valid"]
        self.pad_token_dict = dict(
            coords=0.0,
            chain_ids=-1,
            res_ids=-1,
            masks=False,
            chi_mask=False,
            chi_symm=False,
        )
        self.pad_token_dict["tokenized_sequences"] = AA_TO_IDX["<pad>"]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._get_column(idx)
        item, record = self.get_item(idx)
        record["pdb"] = item["pdb"]
        record["name"] = self.item_uid(record)
        return record

    def get_item(self, idx, packed=True):
        item = self.dataset[idx]
        chain_symbols = item["chains"]
        sequences = [s.replace("?", "X") for s in item["sequence"]]
        sequence = "".join(sequences)
        tokenized_sequences = tokenize(sequence)
        coords = torch.from_numpy(
            unpack_coordinates(item["coords"], sequence)
            if packed
            else item["coords"].copy()
        )
        if not self.include_sidechains:
            coords = coords[:, :4, :].detach().clone()
        record = {
            "tokenized_sequences": tokenized_sequences,
            "coords": coords,
            "chain_symbols": chain_symbols,
        }
        record["sequences"] = sequences

        valid_masks = get_valid_mask(coords)
        chain_ids, res_ids = get_chain_and_res_id_tensors(sequences)
        record.update(
            {
                "chain_ids": chain_ids,
                "res_ids": res_ids,
                "masks": {"valid": valid_masks},
            }
        )
        chi_mask, chi_sym = get_chi_masks(sequences)
        record.update(
            {"chi_mask": chi_mask, "chi_sym": chi_sym}
            | self.save_raw_data(item, record)
        )

        record["coords"] = torch.nan_to_num_(
            nan_to_ca(coords, record["masks"]["valid"]["atom"]), 0
        )
        return item, record

    def get_keep_raw_symbols(self, item):
        return set()

    def save_raw_data(self, item, batch):
        chain_symbols = item["chains"]
        sequences = item["sequence"]
        chain_ids = batch["chain_ids"]
        keep = self.get_keep_raw_symbols(item)
        useqs, ucoords = {}, {}
        for i, symbol in enumerate(chain_symbols):
            if symbol in keep:
                useqs[symbol] = sequences[i]
                ucoords[symbol] = batch["coords"][chain_ids == i].detach().clone()
        return {"raw_sequences": useqs, "raw_coords": ucoords}

    def collate(self, items):
        items = fill_missing_masks(items, self.mask_keys)
        return collate(items, self.pad_token_dict)

    def to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, tuple):
            return tuple(self.to_torch(i) for i in x)
        if isinstance(x, list):
            return [self.to_torch(i) for i in x]
        if isinstance(x, dict):
            return {k: self.to_torch(v) for k, v in x.items()}
        return x


def collate(items, pad_tokens, key=None):
    """
    Collates a list of items into a single data structure.

    Parameters:
        items (list): List of items to collate.
        pad_tokens (dict or any): Dictionary or single value to use for padding tokens.
        key (str, optional): Key to specify a subset of items for collation. Default is None.

    Returns:
        dict or torch.Tensor: Collated items.
    """
    if key is not None:
        pad_tokens = pad_tokens.get(key) if isinstance(pad_tokens, dict) else pad_tokens
        if pad_tokens is None:
            return items
    example = _example(items)
    if isinstance(example, dict):
        return {
            key: collate([d.get(key) for d in items], pad_tokens, key)
            for key in example
        }
    elif isinstance(example, torch.Tensor):
        return collate_tensors(items, pad_tokens)


def collate_tensors(tensors, padding_value):
    """
    Collates a list of tensors into a single tensor. Works for both 1D and higher dimensional tensors.

    Parameters:
        tensors (list): List of tensors to collate.
        padding_value (any): Value to use for padding.
    Returns:
        torch.Tensor: Collated tensor.
    Raises:
        Exception: If tensors have different number of dimensions.
    """
    if len(set(t.ndim for t in tensors)) > 1:
        raise Exception("All tensors must have the same number of dimensions!")
    if tensors[0].ndim == 1:
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    sizes = [max(t.shape[dim] for t in tensors) for dim in range(tensors[0].ndim)]
    padded = []
    for tensor in tensors:
        pad = tuple(
            x for size, shape in zip(sizes, tensor.shape) for x in (size - shape, 0)
        )[::-1]
        padded.append(torch.nn.functional.pad(tensor, pad=pad, value=padding_value))
    return torch.stack(padded)


def fill_missing_masks(items, should_have):
    """
    Fills missing masks in a list of items with zero tensors.

    Parameters:
        items (list): List of items with masks.
        should_have (list): List of mask keys that should be present.

    Returns:
        list: List of items with filled masks.
    """
    mask_pads = [
        {l: torch.zeros_like(x["masks"]["valid"][l]) for l in ("residue", "atom")}
        for x in items
    ]
    for item, mask_pad in zip(items, mask_pads):
        for mask in should_have:
            if mask not in item["masks"]:
                item["masks"][mask] = {}
            for level in ("residue", "atom"):
                if level not in item["masks"][mask]:
                    item["masks"][mask][level] = mask_pad[level]
    return items


def _example(values):
    """Return the first item from the list that is not None"""
    return next((value for value in values if value is not None), None)


def apply_crop_mask(batch, crop_mask):
    """
    Applies a crop mask to each element in a batch, returning a new batch containing only the cropped elements.

    Parameters:
    ----------
    batch : dict
        A dictionary representing a batch of data, where the keys are strings and the values are torch.Tensor.
    crop_mask : ndarray
        A boolean array indicating whether each element should be included in the cropped batch
        (True to keep, False to drop).

    Returns:
    -------
    cropped_batch : dict
        A new dictionary with the same keys as the input `batch`, but containing only the cropped elements.
    """
    return {
        k: (
            apply_crop_mask(v, crop_mask)
            if isinstance(v, dict)
            else v[crop_mask].clone()
        )
        for k, v in batch.items()
    }


def get_region_mask(chain_id, start, end, chain_ids):
    """
    Get a boolean mask for a specific region within a chain.
    Parameters:
        chain_id : int
            The ID of the chain.
        start : int
            The starting index of the region.
        end : int
            The ending index of the region (exclusive).
        chain_ids : torch.Tensor
            1-D array of chain IDs.
    Returns:
        torch.Tensor: Boolean mask indicating the region of interest.
    Examples:
        >>> chain_ids = torch.Tensor([1, 2, 1, 3, 2, 2])
        >>> get_region_mask(1, 1, 4, chain_ids)
        array([False,  True,  True,  True, False, False])
    """
    mask = torch.zeros(len(chain_ids), dtype=bool)
    local_mask = mask[chain_ids == chain_id]
    local_mask[start:end] = True
    mask[chain_ids == chain_id] = local_mask
    return mask


def get_idx_mask(chain_id, idxs, chain_ids):
    """
    Get a boolean mask for a specific indices within a chain.
    Parameters:
        chain_id : int
            The ID of the chain.
        idxs : torch.Tensor
            1-D array of indices.
        chain_ids : torch.Tensor
            1-D array of chain IDs.
    Returns:
        torch.Tensor: Boolean mask indicating the region of interest.
    Examples:
        >>> chain_ids = torch.Tensor([1, 2, 1, 3, 2, 2])
        >>> get_region_mask(1, 1, 4, chain_ids)
        array([False,  True,  True,  True, False, False])
    """
    mask = torch.zeros(len(chain_ids), dtype=bool)
    local_mask = mask[chain_ids == chain_id]
    if idxs is not None:
        local_mask[idxs] = True
    mask[chain_ids == chain_id] = local_mask
    return mask


def get_valid_mask(coords):
    """
    Get boolean mask for valid atoms (no nan coodirnates) and valid residues (all backbone atoms are valid)

    Parameters
        coords : torch.Tensor
            Atom cooridnates.

    Returns:
        (torch.Tensor, torch.Tensor): Tuple of boolean masks indicating the valid residues and valid atoms.
    """
    atom_mask = (~torch.isnan(coords)).any(dim=-1)
    res_mask = atom_mask[:, :4].any(dim=-1)
    return {"atom": atom_mask, "residue": res_mask}


def get_chain_and_res_id_tensors(sequences):
    """
    Returns tensors representing the chain IDs and residue IDs for sequences in a list.

    Parameters:
    ----------
    sequences : list
        A list of residue sequences, for each chain in the structure.

    Returns:
    -------
    chain_ids : torch.Tensor
        A int64 1D tensor representing the chain IDs for each residue in the input sequences.
        All residues of the first chain will get id 0, of the second chain id 1 and so on.
    res_ids : torch.Tensor
        A 1D tensor representing the residue IDs for each residue in the input sequences.
        Residues within each chain will get numbered 0..len(seqeunce).
    """
    chain_ids = torch.hstack(
        [torch.full((len(s),), i, dtype=int) for i, s in enumerate(sequences)]
    )
    res_ids = torch.hstack(
        [torch.arange(size) for size in torch.unique(chain_ids, return_counts=True)[1]]
    )
    return chain_ids, res_ids
