from __future__ import annotations
import math
from enum import Enum
from typing import Tuple, List, Union, Optional, Dict, Any

import torch
from torch import Tensor
from einops import rearrange, repeat
from transformers import AutoConfig
from transformers.tokenization_utils_base import BatchEncoding

from igdesign.features.feature_constants import ATOM_TY_TO_ATOM_IDX


get_max_val = lambda x: torch.finfo(x.dtype).max
get_min_val = lambda x: torch.finfo(x.dtype).min
PI = math.pi
TRUE, FALSE = torch.ones(1).bool(), torch.zeros(1).bool()
cos_max, cos_min = (1 - 1e-9), -(1 - 1e-9)
min_norm_clamp = 1e-7


def exists(x: Any) -> bool:
    """Returns True if and only if x is not None"""
    return x is not None


def default(x: Any, y: Any) -> Any:
    """Returns x if x exists, otherwise y"""
    return x if exists(x) else y


def safe_to_device(x: Optional[Union[Dict, Tensor, List]], device) -> Any:
    """Places tensor objects in x on specified device"""
    if isinstance(x, dict):
        return {k: safe_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [safe_to_device(v, device) for v in x]
    # base case
    return x.to(device) if torch.is_tensor(x) else x


def coords_to_rel_coords(coords: Tensor) -> Tensor:
    """
    Map from coordinates coords[b,i,:] = x_i in R^k to relative coordinates
    rel_coords[b,i,j,:] = x_j-x_i in R^k
    """
    return rearrange(coords, "b n c -> b () n c") - rearrange(
        coords, "b n c -> b n () c"
    )


def pair_mask_from_res_mask(res_mask: Tensor) -> Tensor:
    """
    Convert residue mask to pair mask
    out[...,i,j] = res_mask[...,i] or res_mask[...,j]
    """
    mask = ~res_mask.bool()
    return ~torch.einsum("... i, ... j -> ... i j", mask, mask)


def valid_dihedral_mask(valid_res_mask: Tensor, res_ids: Optional[Tensor]) -> Tensor:
    """
    Get backbone dihedral feature mask
    mask[...,i] = True iff  bb-dihedral can be computed for residue i
    """
    mask = valid_res_mask.clone()
    seq_gaps = (res_ids[..., 1:] - res_ids[..., :-1]) != 1 if exists(res_ids) else None
    # A sequence gap invalidates a dihedral for the residues on either side
    mask[..., 1:] = (
        torch.logical_and(~seq_gaps, mask[..., 1:])
        if exists(seq_gaps)
        else mask[..., 1:]
    )
    mask[..., :-1] = (
        torch.logical_and(~seq_gaps, mask[..., :-1])
        if exists(seq_gaps)
        else mask[..., :-1]
    )
    # Residues on either side of dihedral must be valid
    mask[..., :-1] = torch.logical_and(valid_res_mask[..., 1:], mask[..., :-1])
    mask[..., 1:] = torch.logical_and(valid_res_mask[..., :-1], mask[..., 1:])
    # Dihedrals for first and last residues are never valid
    mask[..., 0] = mask[..., -1] = False
    return mask


def get_bipartite_mask(
    mask: Tensor,
    S_indices: Tensor,
    T_indices: Optional[Tensor] = None,
    fill: Any = True,
) -> Tensor:
    """Mask all (i,j) such that (i in S and j in T) or (j in S and i in T)"""
    T_indices = default(T_indices, S_indices)
    rep_T = repeat(T_indices, "i -> i m", m=S_indices.numel())
    rep_S = repeat(S_indices, "i -> i m", m=T_indices.numel())
    mask[S_indices, rep_T] = fill
    mask[T_indices, rep_S] = fill
    return mask


def get_partition_mask(
    n: int,
    partition: List[Tensor],
    part_adjs: Optional[Tensor] = None,
) -> Tensor:
    """
    Mask edges crossing between components of partition.

    e.g. if partition = [[1,2],[6], [8]], then edges according to the
    adjacency lists :
        1 : (6,7,8)
        2:  (6,7,8)
        6:  (1,2,8)
        8 : (1,2,6)
    will be masked (in the undirected sense).

    :param n: size of underlying graph
    :param partition: partition of the vertices (NOTE: does not strictly
    need to be a partition, can consist of any subsets of vertices)
    :param part_adjs: adjacency matrix for subsets of partition
    :return: adjacency mask with partition edges set to "True", and
    all other edges set to "False".
    """
    assert len(partition) > 0
    device = partition[0].device
    part_adjs = default(part_adjs, torch.ones(n, n, device=device).bool())
    mask = torch.zeros(n, n, device=device).bool()
    for i in range(len(partition)):
        X = partition[i]
        for j in range(i + 1, len(partition)):  # noqa
            if part_adjs[i, j]:
                Y = partition[j]
                mask = get_bipartite_mask(mask, X, Y, fill=TRUE.to(device))
    return mask


def get_inter_chain_mask(
    chain_lens: List[int], batch_size: Optional[int] = None
) -> Tensor:
    """Indicates whether residues i and j are in different chains, given chain lengths"""
    bounds = torch.cumsum(torch.tensor([0] + chain_lens), dim=0)
    partition = [torch.arange(bounds[i], bounds[i + 1]) for i in range(len(chain_lens))]
    mask = get_partition_mask(n=bounds[-1], partition=partition)
    if exists(batch_size):
        return repeat(mask, "i j -> b i j", b=batch_size)
    return mask


def get_inter_chain_contacts(
    coords: Tensor,
    chain_ids: Tensor,
    atom_pairs: List[Tuple[str, str]],
    max_dist: float = 10.0,
):
    (b, n), device = coords.shape[:2], coords.device
    # Create mask indicating which residues belong to same chain
    chain_diff = rearrange(chain_ids, "b n -> b n ()") - rearrange(
        chain_ids, "b n -> b () n"
    )
    same_chain_mask = chain_diff == 0

    # Get pairwise distances
    dists = torch.zeros(b, n, n, len(atom_pairs), device=device)
    for idx, (a1, a2) in enumerate(atom_pairs):
        c1, c2 = map(lambda atom: coords[:, :, ATOM_TY_TO_ATOM_IDX[atom]], (a1, a2))
        dists[..., idx] = torch.cdist(c1, c2)
    dists, _ = torch.min(dists, dim=-1)
    # Mask out pairs that are (1) within the same chain, and (2) beyond threshold distance
    dists[same_chain_mask] = max_dist + 1
    return dists <= max_dist


def string_encode(mapping: Dict[str, int], tokens: str) -> Tensor:
    """
    Encodes a string (or list of strings) according to the given mapping

    Parameters
        mapping: map from string to integer defining encoding
        tokens: string to encode
    Returns
        encoded tokens according to given mapping
    """
    return torch.tensor([mapping[token] for token in tokens], dtype=torch.long)


def fourier_encode(x: Tensor, num_encodings=4, include_self=True) -> Tensor:
    """
    Applies fourier encoding (sin + cos scaled by freq.) to input x

    :param x: tensor to apply encoding to
    :param num_encodings: number of frequencies to encode for (1,...1/2**(num_encodings-1))
    :param include_self: whether to append x[...-1] to encodings
    :return: fourier encoding of x
    """
    trailing_one = x.shape[-1] == 1
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze(-2) if trailing_one else x


def bin_encode(data: Tensor, bins: Tensor):
    """
    Assigns each value in data to
    :param data: the data to apply bin encoding to
    :param bins: description of bin positions to encode into
        [(bins[i],bins[i+1])] is used to define each position.
    :return: bin index of each value in input data
    """
    assert torch.min(data) >= bins[0] and torch.max(data) < bins[-1], (
        f"incorrect bins, got min/max of data: ({torch.min(data)},{torch.max(data)})\n"
        f"but bin min/max = ({bins[0]},{bins[-1]}])"
    )
    binned_data = -torch.ones_like(data)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = torch.logical_and(data >= low, data < high)  # noqa
        binned_data[mask] = i
    return binned_data.long()


class TrRosettaOrientationType(Enum):
    """Type enum for trRosetta dihedral"""

    PHI = ["N", "CA", "CB", "CB"]
    PSI = ["CA", "CB", "CB"]
    OMEGA = ["CA", "CB", "CB", "CA"]


def get_tr_rosetta_orientation_mat(
    N: Tensor, CA: Tensor, CB: Tensor, ori_type: TrRosettaOrientationType
) -> Tensor:
    """
    Gets trRosetta dihedral matrix for the given coordinates
    :param N: backbone Nitrogen coordinates - shape (b,n,3) or (n,3)
    :param CA: backbone Nitrogen coordinates - shape (b,n,3) or (n,3)
    :param CB: backbone Nitrogen coordinates - shape (b,n,3) or (n,3)
    :param ori_type: trRosetta dihedral type to compute
    :return: dihedral matrix with shape (b,n,n,3) or (n,n,3)
    """
    if ori_type == TrRosettaOrientationType.PSI:
        mat = unsigned_angle_all([CA, CB, CB])
    elif ori_type == TrRosettaOrientationType.OMEGA:
        mat = signed_dihedral_all_12([CA, CB, CB, CA])
    elif ori_type == TrRosettaOrientationType.PHI:
        mat = signed_dihedral_all_123([N, CA, CB, CB])
    else:
        raise Exception(f"dihedral type {ori_type} not accepted")
    return mat


def get_tr_rosetta_orientation_mats(
    N: Tensor, CA: Tensor, CB: Tensor
) -> Tuple[Tensor, ...]:
    phi = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.PHI)
    psi = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.PSI)
    omega = get_tr_rosetta_orientation_mat(N, CA, CB, TrRosettaOrientationType.OMEGA)
    return phi, psi, omega


def get_bb_dihedral(N: Tensor, CA: Tensor, C: Tensor) -> Tuple[Tensor, ...]:
    """
    Gets backbone dihedrals for
    :param N: (n,3) or (b,n,3) tensor of backbone Nitrogen coordinates
    :param CA: (n,3) or (b,n,3) tensor of backbone C-alpha coordinates
    :param C: (n,3) or (b,n,3) tensor of backbone Carbon coordinates
    :return: phi, psi, and omega dihedrals angles (each of shape (n,) or (b,n))
    """
    assert all([len(N.shape) == len(x.shape) for x in (CA, C)])
    squeeze = len(N.shape) == 2
    N, CA, C = map(lambda x: x.unsqueeze(0), (N, CA, C)) if squeeze else (N, CA, C)
    b, n = N.shape[:2]
    phi, psi, omega = [torch.zeros(b, n, device=N.device) for _ in range(3)]
    phi[:, 1:] = signed_dihedral_4([C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:]])
    psi[:, :-1] = signed_dihedral_4([N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:]])
    omega[:, :-1] = signed_dihedral_4([CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:]])
    return (
        map(lambda x: x.squeeze(0), (phi, psi, omega)) if squeeze else (phi, psi, omega)
    )


def signed_dihedral_4(
    ps: Union[Tensor, List[Tensor]], return_mask=False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Computes (signed) dihedral angle of input points.

     works for batched and unbatched point lists

    :param ps: a list of four tensors of points. dihedral angle between
    ps[0,i],ps[1,i],ps[2,i], and ps[3,i] will be ith entry of output.
    :param return_mask: whether to return a mask indicating where dihedral
    computation may have had precision errors.

    :returns : list of dihedral angles
    """
    p0, p1, p2, p3 = ps.unbind(dim=-3) if torch.is_tensor(ps) else ps
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    mask = torch.norm(b1, dim=-1) > 1e-7
    b1 = torch.clamp_min(b1, 1e-6)
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    res = torch.atan2(y, x)
    return res if not return_mask else (res, mask)


def signed_dihedral_all_12(ps: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    Computes signed dihedral of points taking p2-p1 as dihedral axis
    :param ps:
    :return:
    """
    p0, p1, p2, p3 = ps.unbind(dim=-3) if torch.is_tensor(ps) else ps
    b0, b1, b2 = p0 - p1, p2.unsqueeze(-3) - p1.unsqueeze(-2), p3 - p2
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True).clamp_min(min_norm_clamp)
    v = b0.unsqueeze(-2) - torch.sum(b0.unsqueeze(-2) * b1, dim=-1, keepdim=True) * b1
    w = b2.unsqueeze(-3) - torch.sum(b2.unsqueeze(-3) * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    return torch.atan2(y, x)


def signed_dihedral_all_123(ps: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    :param ps:
    :return:
    """
    p0, p1, p2, p3 = ps.unbind(dim=-3) if torch.is_tensor(ps) else ps
    b0, b1, b2 = p0 - p1, p2 - p1, p3.unsqueeze(-3) - p2.unsqueeze(-2)
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True).clamp_min(min_norm_clamp)
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1.unsqueeze(-2), dim=-1, keepdim=True) * b1.unsqueeze(-2)
    x = torch.sum(v.unsqueeze(-2) * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1).unsqueeze(-2) * w, dim=-1)
    ret = torch.atan2(y, x)
    return ret


def unsigned_angle_all(ps: List[Tensor]) -> Tensor:
    """
    Retrieves a matrix of (unsigned) angles between input points

    returns: a matrix M where M[i,j] is the angle btwn the lines formed
    by ps0[i],ps1[i] and ps[1,i],ps[2,j].
    """
    p0, p1, p2 = ps.unbind(dim=-3) if torch.is_tensor(ps) else ps
    b01, b12 = p0 - p1, p2.unsqueeze(-3) - p1.unsqueeze(-2)
    M = b01.unsqueeze(-2) * b12
    n01, n12 = torch.norm(b01, dim=-1, keepdim=True), torch.norm(b12, dim=-1)
    prods = torch.clamp_min(n01 * n12, min_norm_clamp)
    cos_theta = torch.sum(M, dim=-1) / prods
    cos_theta[cos_theta < cos_min] = cos_min
    cos_theta[cos_theta > cos_max] = cos_max
    return torch.acos(cos_theta)


def impute_beta_carbon(coords: Tensor, mask: Tensor) -> Tensor:
    """
    Imputes coordinates of beta carbon from tensor of residue coordinates
    :param coords: shape (b,n,*,3) where dim=1 is N,CA,C,... coordinates.
    :param mask: shape(b,n) tensor
    :return: imputed CB coordinates for each residue
    """
    bb_coords = coords.clone()
    bb_coords[~mask] = torch.randn_like(bb_coords[~mask])
    bb_coords = rearrange(coords, "b n a c -> b a n c")
    N, CA, C, *_ = bb_coords.unbind(dim=-3)
    n, c = N - CA, C - CA
    n_cross_c = torch.cross(n, c, dim=-1)
    t1 = math.sqrt(1 / 3) * (n_cross_c / torch.norm(n_cross_c, dim=-1, keepdim=True))
    n_plus_c = n + c
    t2 = math.sqrt(2 / 3) * (n_plus_c / torch.norm(n_plus_c, dim=-1, keepdim=True))
    imputed_points = CA + (t1 + t2)
    imputed_points[~mask] = CA[~mask]
    return imputed_points


def get_dim_out(pretrained_model_name_or_path: str):
    dim_out = AutoConfig.from_pretrained(pretrained_model_name_or_path).hidden_size
    return dim_out


def cleanup_tokenizer_init(pretrained_model_name_or_path, tok_kwargs):
    """
    Remove the special tokens from the tokenizer's vocab
    """
    if "progen" in pretrained_model_name_or_path:
        register_progen()  # Register ProGen classes for AutoModel and AutoConfig
        tok_kwargs |= {
            "tokenizer_class": "ProGenTokenizer"
        }  # GPT2 -> ProGenTokenizer for auto-tokenizer loading
    elif "oas" in pretrained_model_name_or_path:
        tok_kwargs |= {"use_fast": False, "model_max_length": 512}
        # RoBERTa-OAS tokenizer is bugged for fast tokenizer masking


def match_aa_dict_to_tokenizer(aa_dict, tokenizer):
    """
    Replace the special symbols in the AA dictionary with the ones used in the tokenizer
        - e.g. Progen uses <|mask|> and <|pad|> instead of <mask> and <pad>
    """
    old2new = {
        getattr(aa_dict, k.replace("token", "word"), None): v
        for k, v in tokenizer.special_tokens_map.items()
        if getattr(aa_dict, k.replace("token", "word"), None) is not None
    }

    def replace_all(text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    aa_dict.symbols = list(map(lambda x: replace_all(x, old2new), aa_dict.symbols))
    return aa_dict


def mask_attention_mask(encoded_input: BatchEncoding, tokenizer) -> Tensor:
    """
    Mask out masked and padded tokens in attention_mask
        - This is necessary for the LM to not attend to these tokens
            in case tokenizer doesn't do it automatically
    :param encoded_input: BatchEncoding object from tokenizer
    :return: encoded_input with masked and padded tokens masked out in attention mask
    """
    encoded_input["attention_mask"] *= torch.where(
        (encoded_input["input_ids"] == tokenizer.mask_token_id)
        | (encoded_input["input_ids"] == tokenizer.pad_token_id),
        0,
        1,
    )

    return encoded_input


def mean_pooling(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Mean Pooling - Take attention mask into account for correct averaging
    :param last_hidden_state: Last layer hidden-state of the first token of the sequence (classification token)
    :param attention_mask: Attention mask of the input sequence

    :return: Mean-pooled sentence embeddings
    """
    token_embeddings = last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
