from typing import List, Union
from typeguard import typechecked

import torch
from torchtyping import TensorType
from einops import repeat


def nan_to_ca(coords, mask, batched=False):
    """Sets coordinates that have nan values to Ca coordinates of the same residue"""
    natoms = coords.shape[-2]
    pattern = "b n c -> b n a c" if batched else "n c -> n a c"
    ca = repeat(coords[..., 1, :], pattern, a=natoms).clone()
    coords[~mask] = ca[~mask]
    return coords


@typechecked
def masked_align_structures(
    mobile: Union[
        TensorType["batch", "seq_len", "atoms", 3], TensorType["batch", "seq_len", 3]
    ],
    target: Union[
        TensorType["batch", "seq_len", "atoms", 3], TensorType["batch", "seq_len", 3]
    ],
    mask: Union[
        TensorType["batch", "seq_len", "atoms"], TensorType["batch", "seq_len"]
    ] = None,
    atom_selection: List = [0, 1, 2, 3],  # defaults to all backbone atoms
    in_place: bool = False,
):
    """
    Returns matrices from ``mobile`` aligned to ``target`` using rotation and translation.
    Only the atoms in atom_selection are used to perform the alignment

    Parameters
    ----------
    mobile: TensorType["batch", "seq_len", "atoms", "dim"]
        Batch of matrices that should be aligned
    target: TensorType["batch", "seq_len", "atoms", "dim"]
        Reference structure
    mask: TensorType["batch", "seq_len", "atoms"]
        Mask for which residues/atoms should be ignored for alignment. NaNs and
        atoms outside atom_selection are already ignored
    atom_selection:
        Which atoms from each residue to use for alignment. e.g. [1] for CAs,
        [0,1,2,3] for backbone. Note same indices used for all atoms.
    Atom dimension optional for all tensors

    Returns
    -------
    TensorType["batch", "seq_len", "atoms", 3]
        Aligned matrices in ``mobile`` to respective references in ``target``.
    """
    assert (
        mobile.shape == target.shape
    ), "Mobile and target structures must have same shape"
    if mask is not None:
        assert mask.dim() < mobile.dim(), "Mask should have fewer dims than structures"
    if in_place:
        assert (
            mobile.requires_grad == False
        ), "set requires_grad=False for in place alignment"
    # Select atoms and flatten
    if mobile.dim() == 4:
        mobile_sel, target_sel = map(
            lambda x: x[:, :, atom_selection, :].flatten(1, 2), [mobile, target]
        )
    else:
        mobile_sel = mobile
        target_sel = target

    # Reshape mask
    if mask is None:
        atom_mask = torch.ones(*mobile_sel.shape[:2], device=mobile.device)
    elif mask.dim() == 2:
        # Add atom axis if necessary
        if mobile.dim() == 4:
            atom_mask = (
                mask.unsqueeze(-1).expand(-1, -1, len(atom_selection)).flatten(1, 2)
            )
        else:
            atom_mask = mask
    elif mask.dim() == 3:
        # If mask has an atom dimension, expect same atoms as in mobile and target to be present
        atom_mask = mask[:, :, atom_selection].flatten(1, 2)
    atom_mask = (
        atom_mask.unsqueeze(-1).bool() & ~mobile_sel.isnan()
    ).int()  # add spatial dims axis

    # Center mobile and target, average over atoms
    mobile_mean, target_mean = map(
        lambda x: (x * atom_mask).nansum(dim=1, keepdim=True)
        / atom_mask.nansum(dim=1, keepdim=True),
        [mobile_sel, target_sel],
    )
    mobile_sel, target_sel = mobile_sel - mobile_mean, target_sel - target_mean

    # Perform alignment
    mobile_sel, target_sel = map(torch.nan_to_num, [mobile_sel, target_sel])
    R = kabsch(mobile_sel * atom_mask, target_sel * atom_mask)
    if in_place:
        mobile -= mobile_mean.unsqueeze(2)
        tmp = torch.clone(mobile)
        torch.matmul(tmp, R, out=mobile)
        mobile += target_mean
    else:
        return ((mobile.flatten(1, -2) - mobile_mean) @ R + target_mean).view(
            *mobile.shape
        )
