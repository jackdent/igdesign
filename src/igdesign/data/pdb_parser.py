from typing import Dict, Union
from string import ascii_letters, digits

import numpy as np
import pandas as pd
from Bio import PDB

from igdesign.tokenization import IUPAC_CODES, ATOM_LIST, BACKBONE_ATOM_LIST


def three2one(residue):
    residue = residue if residue.capitalize() != "Unk" else "Xaa"
    return IUPAC_CODES.get(residue.capitalize(), "")


def sort_inner_keys(dictonary):
    """
    Takes a dict of dicts; Each inner dict is replaced with a version where keys are sorted,
    but values kept in place: {"A": {2: "a", 1: "b"}} -> {"A": {1: "a", 2: "b"}}.
    """
    return {
        k: {k2: v[k1] for k1, k2 in zip(v.keys(), sorted(v.keys()))}
        for k, v in dictonary.items()
    }


def encode_rsid(vals):
    """
    Turns a dict of rs_id values into a uint16 ndarray len x 3,
    with the first column being the numeric part of the rsid,
    second column being the letter part (32 or ord(" ") for no letter),
    and the third column being the position in the retured arrays.
    """
    p_array = np.empty((len(vals), 3), dtype=np.int16)
    for i, (rsid, idx) in enumerate(vals.items()):
        p_array[i, 0] = int(rsid.rstrip(ascii_letters))
        p_array[i, 1] = ord(rsid.lstrip(digits + "-") or " ")
        p_array[i, 2] = idx
    return p_array


def decode_rsid(vals):
    """Turns the ndarray from encode_rsids back into a dict."""
    return {str(x[0]) + chr(x[1]).replace(" ", ""): x[2] for x in vals}


class PDBParserAnnotator:
    """
    PDBParserAnnotator transforms and stores
    pdb chain information (sequence, coordinates etc) and
    make it ingestable by other classes/functions.

    Attributes:
        atoms (List): List of target atoms for storing information
    """

    def __init__(self, verbal_flag=True):
        self.verbal_flag = verbal_flag

    def parse_PDB(self, pdb_file, chain, backbone_only=False, skip_check=False):
        """
        Parameters:
                pdb_file (String): PDB filename
                chain (String): chain name
        Returns:
                Dictionary record with sequence, coords
        """
        atom_list = BACKBONE_ATOM_LIST if backbone_only else ATOM_LIST

        if chain is np.NaN or chain == "NA":
            return {}
        if self.verbal_flag:
            print(" Processing ", str(pdb_file).split("/")[-1])
        xyz, seq = {}, {}
        with open(pdb_file, "rb") as f:
            for line in f.readlines():
                line = line.decode("utf-8", "ignore").rstrip()

                if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
                    line = line.replace("HETATM", "ATOM  ")
                    line = line.replace("MSE", "MET")

                if line[:4] == "ATOM":
                    ch = line[21:22]
                    if ch == chain or chain is None:
                        atom = line[12 : 12 + 4].strip()
                        resi = line[17 : 17 + 3]
                        resn = line[22 : 22 + 5].strip()
                        x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                        if resn[-1].isalpha():
                            resa, resn = resn[-1], int(resn[:-1]) - 1
                        else:
                            resa, resn = "", int(resn) - 1

                        if resn not in xyz:
                            xyz[resn] = {}
                        if resa not in xyz[resn]:
                            xyz[resn][resa] = {}
                        if resn not in seq:
                            seq[resn] = {}
                        if resa not in seq[resn]:
                            seq[resn][resa] = resi

                        if atom not in xyz[resn][resa]:
                            xyz[resn][resa][atom] = np.array([x, y, z])

        # Get missing residues and update seq/xyz dicts
        missing_res = self._parse_missing_residues(pdb_file, chain)
        for resn in missing_res:
            if resn not in seq:
                seq[resn] = {}
                xyz[resn] = {}

            for k in missing_res[resn]:
                assert (
                    k not in seq[resn]
                ), f"Missing residue overlaps with ATOM records at {resn + 1}{k} in chain {chain} of {pdb_file}"
                seq[resn][k] = missing_res[resn][k]

                xyz[resn][k] = {}
                for atom in atom_list:
                    xyz[resn][k][atom] = np.full(3, np.nan)

        if chain is not None and not skip_check:
            seqres_seq = self.get_seqres_seq(
                pdb_file, chain
            )  # only used for validating extracted seqs

        final_seq = "".join(
            three2one(seq[resn][k]) for resn in sorted(seq) for k in sorted(seq[resn])
        )
        # Compare final seq with SEQRES records
        if not skip_check and chain is not None and final_seq != seqres_seq:
            seq = sort_inner_keys(seq)
            xyz = sort_inner_keys(xyz)
            final_seq = "".join(
                three2one(seq[resn][k])
                for resn in sorted(seq)
                for k in sorted(seq[resn])
            )

            if final_seq != seqres_seq:
                error = f"Parsed does not match SEQRES records for chain {chain} in {pdb_file}.\n"
                error += f"Is Biopython fully updated?\n Parsed: {final_seq}\n SEQRES: {seqres_seq}"
                raise ValueError(error)

        # Convert to numpy arrays
        xyz_ = []
        MISSING = np.full(3, np.nan)
        for resn in sorted(seq):
            for k in sorted(seq[resn]):
                for atom in atom_list:
                    xyz_.append(xyz[resn][k].get(atom, MISSING))

        valid_resn = np.array(sorted(xyz.keys()))
        final_coords = np.array(xyz_).reshape(-1, len(atom_list), 3)
        if final_coords.shape[0] != len(final_seq):
            raise ValueError("coordinates and sequence length must match")

        if chain is not None and not skip_check:
            # Compare final seq with SEQRES records
            if final_seq != seqres_seq:
                msg = f"Final seq parsed does not match SEQRES records for chain {chain} in {pdb_file}. "
                msg += "Is Biopython fully updated?"
                raise ValueError(msg)

        flat_seq = [
            f"{k_out + 1}{k_in}" for k_out in sorted(seq) for k_in in sorted(seq[k_out])
        ]
        record = {
            "coords": final_coords,
            "sequence": final_seq,
            "valid_keys": valid_resn,
            "res_ids": encode_rsid({sid: i for i, sid in enumerate(flat_seq)}),
            "atoms": atom_list,
        }
        return record

    def get_seqres_seq(self, pdb_file: str, chain_id: str) -> Union[str, float]:
        """
        get_seqres_seq gets the sequence corresponding to the chain_id in pdb_file using
        the SEQRES lines in the PDB file header rather than using the atoms of the 3D structure.
        Residues that appear in the SEQRES lines but not the ATOM records in the 3D structure are
        missing residues.

        If chain_id is np.nan, return np.nan.

        Reference: https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences-and-the-pdb-format

        Parameters
        ----------
        pdb_file : string
            Input pdb file to extract sequence from
        chain_id : string
            Chain ID to extract sequence for

        Returns
        -------
        string
            SEQRES sequence for the chain in the PDB file
        """
        if pd.isnull(chain_id):
            return np.nan

        seqres_entries = []
        with open(pdb_file, "rb") as f:
            for line in f.readlines():
                line = line.decode("utf-8", "ignore").rstrip()
                if line[:6] != "SEQRES":
                    continue
                if line[11] != chain_id:
                    continue
                idx = int(line[6:10])
                seqres_entries.append((idx, line[19:].split()))

        assert (
            len(seqres_entries) != 0
        ), f"SEQRES info for chain {chain_id} not found in {pdb_file}"

        # Make sure seqres_entries are formatted properly
        seqres_entries = sorted(seqres_entries, key=lambda x: x[0])

        residues_chain = []
        for i, (idx, residues) in enumerate(seqres_entries):
            residues_chain += residues
            if i == 0:
                assert idx == 1, f"First SEQRES entry index should be 1 in {pdb_file}"
                idx_prev = idx
                residues_prev = residues
                continue

            # Verify SEQRES entry index
            assert (
                idx == idx_prev + 1
            ), f"Expected successive SEQRES entry indices to increment by 1 in {pdb_file}"
            idx_prev = idx

            # Verify number of residues on each line (besides the last line)
            if i == len(seqres_entries) - 1:
                continue
            assert len(residues) == len(
                residues_prev
            ), f"Expected SEQRES to have the same number of residues on each line besides the last in {pdb_file}"
            residues_prev = residues

        # Convert MSE to MET
        for i in range(len(residues_chain)):
            if residues_chain[i] == "MSE":
                residues_chain[i] = "MET"

        residues = "".join([three2one(x) for x in residues_chain])
        return residues

    def _parse_missing_residues(
        self, pdb_file: str, chain_id: str
    ) -> Dict[int, Dict[str, str]]:
        missing_residue_entries = PDB.parse_pdb_header(pdb_file)["missing_residues"]

        missing_res = {}
        for entry in missing_residue_entries:
            if entry["chain"] != chain_id:
                continue
            res = (
                entry["res_name"] if entry["res_name"].capitalize() != "Unk" else "Xaa"
            )

            if res.capitalize() not in IUPAC_CODES:
                # Skip non-standard residues except MSE
                if res == "MSE":
                    res = "MET"
                else:
                    continue

            resn = int(entry["ssseq"]) - 1
            resa = entry["insertion"] if entry["insertion"] is not None else ""

            if resn not in missing_res:
                missing_res[resn] = {}

            assert (
                resa not in missing_res[resn]
            ), f"Duplicate residue found in {pdb_file} for {resn + 1}{resa} in chain {chain_id} of {pdb_file}"
            missing_res[resn][resa] = res

        return missing_res
