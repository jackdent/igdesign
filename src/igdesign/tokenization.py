import numpy as np
import torch


base_tokens = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
]

special_symbols = ["<mask>"]

AMINO_ACID_LIST = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    "#",
    "-",
]

tokens = base_tokens + special_symbols + AMINO_ACID_LIST

AA_TO_IDX = {x: i for i, x in enumerate(tokens)}
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}

BACKBONE_ATOM_LIST = ["N", "CA", "C", "O"]
SIDE_CHAIN_ATOM_LIST = [
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
ATOM_LIST = BACKBONE_ATOM_LIST + SIDE_CHAIN_ATOM_LIST


IUPAC_CODES = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Val": "V",
    "Tyr": "Y",
    "Asx": "X",  # "B", This is to avoid adding atom 37
    "Sec": "X",  # "U", This is to avoid adding atom 37
    "Xaa": "X",
    "Glx": "X",  # "Z", This is to avoid adding atom 37
}


def tokenize(sequence):
    idx_list = [AA_TO_IDX[x] for x in sequence]
    return torch.tensor(idx_list)


def detokenize(tokens):
    return


# Defining residue defaults
NDIH = 8
NCHI = 4
RES_RIGID_GROUPS = {}
RES_RIGID_INDICES14 = {}
RES_RIGID_INDICES37 = {}
RES_RIGID_DEFAULT_ROT = {}
RES_RIGID_DEFAULT_TRANS = {}
RES_RIGID_DEFAULT_COORDS = {}
ATOM_SWAP_MATRICES14 = {}
ATOM_SWAP_MATRICES37 = {}

NATOM_RIGID = 8  # max number of atoms in a rigid group (fused ring trp)
RIGID_ATOM_COORDS = torch.zeros(len(tokens), NDIH, NATOM_RIGID, 3)
RES_RIGID_ROT_TENSOR = torch.zeros(len(tokens), NDIH, 3, 3)
RES_RIGID_TRANS_TENSOR = torch.zeros(len(tokens), NDIH, 3)
RIGID_ATOM_IDX14 = torch.ones(len(tokens), NDIH, NATOM_RIGID).long() * 14
RIGID_ATOM_IDX37 = torch.ones(len(tokens), NDIH, NATOM_RIGID).long() * 37
ATOM_SWAP_TENSOR14 = torch.stack([torch.eye(14)] * len(tokens), dim=0)
ATOM_SWAP_TENSOR37 = torch.stack([torch.eye(37)] * len(tokens), dim=0)
CHI_ATOM_INDICES14 = torch.zeros(len(tokens), NCHI, 4).long()
CHI_ATOM_INDICES37 = torch.zeros(len(tokens), NCHI, 4).long()
CHI_MASK_TENSOR = torch.zeros(len(tokens), NCHI)
CHI_SYMM_TENSOR = torch.zeros(len(tokens), NCHI)
ATOM_MASK_TENSOR14 = torch.zeros(len(tokens), 14)
ATOM_MASK_TENSOR37 = torch.zeros(len(tokens), 37)
ATOM_MASK_TENSOR14[:, :4] = 1.0
ATOM_MASK_TENSOR37[:, :4] = 1.0


def get_chi_mask_batch(tokens):
    """
    [tokens] = [batch, residues]
    CHI_MASK_TENSOR = [AAs, chi angles]
    returns
    dih mask = [batch, residues, chi angles]
    """
    device = tokens.device
    mask = CHI_MASK_TENSOR.to(device)
    res_indices = tokens[(...,) + (None,) * 2].expand(*tokens.shape, 1, NCHI).long()
    mask = mask[(None,) * 2 + (...,)].expand(*tokens.shape, *mask.shape).to(device)
    return mask.gather(dim=2, index=res_indices).squeeze()


def get_chi_symm_batch(tokens):
    """
    [tokens] = [batch, residues]
    CHI_SYMM_TENSOR = [AAs, chi angles]
    returns
    dih mask = [batch, residues, chi angles]
    """
    device = tokens.device
    mask = CHI_SYMM_TENSOR.to(device)
    res_indices = tokens[(...,) + (None,) * 2].expand(*tokens.shape, 1, NCHI).long()
    mask = mask[(None,) * 2 + (...,)].expand(*tokens.shape, *mask.shape)
    return mask.gather(dim=2, index=res_indices).squeeze()


RES_ATOM37_IDX = {
    "A": np.array([0, 1, 2, 3, 4]),
    "R": np.array([0, 1, 2, 3, 4, 5, 11, 23, 32, 29, 30]),
    "N": np.array([0, 1, 2, 3, 4, 5, 16, 15]),
    "D": np.array([0, 1, 2, 3, 4, 5, 16, 17]),
    "C": np.array([0, 1, 2, 3, 4, 10]),
    "Q": np.array([0, 1, 2, 3, 4, 5, 11, 26, 25]),
    "E": np.array([0, 1, 2, 3, 4, 5, 11, 26, 27]),
    "G": np.array([0, 1, 2, 3]),
    "H": np.array([0, 1, 2, 3, 4, 5, 14, 13, 20, 25]),
    "I": np.array([0, 1, 2, 3, 4, 6, 7, 12]),
    "L": np.array([0, 1, 2, 3, 4, 5, 12, 13]),
    "K": np.array([0, 1, 2, 3, 4, 5, 11, 19, 35]),
    "M": np.array([0, 1, 2, 3, 4, 5, 18, 19]),
    "F": np.array([0, 1, 2, 3, 4, 5, 12, 13, 20, 21, 32]),
    "P": np.array([0, 1, 2, 3, 4, 5, 11]),
    "S": np.array([0, 1, 2, 3, 4, 8]),
    "T": np.array([0, 1, 2, 3, 4, 9, 7]),
    "W": np.array([0, 1, 2, 3, 4, 5, 12, 13, 24, 21, 22, 33, 34, 28]),
    "Y": np.array([0, 1, 2, 3, 4, 5, 12, 13, 20, 21, 32, 31]),
    "V": np.array([0, 1, 2, 3, 4, 6, 7]),
    "X": np.array([0, 1, 2, 3]),
}
