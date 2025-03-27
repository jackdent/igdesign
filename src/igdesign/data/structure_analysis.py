import subprocess
import tempfile
from pathlib import Path
from typing import TextIO

import cytoolz as ct
import numpy as np

# Ref Anarci: schemes.py IMGT
IMGT_BOUNDARIES = {
    "imgt_fwr1": (1, 26),
    "imgt_cdr1": (27, 38),
    "imgt_fwr2": (39, 55),
    "imgt_cdr2": (56, 65),
    "imgt_fwr3": (66, 104),
    "imgt_cdr3": (105, 117),
    "imgt_fwr4": (118, 128),
}

PDB_ATOM = "pdb-atom"
PROTEIN_X = "X"
PROTEIN_U = "U"
PROTEIN_G = "G"


def parse_record(fp: TextIO):
    """
    Parses one record from the ANARCI output file.
    Currently assumes human data and IMGT numbering.

    Params:
        fp: File pointer to the ANARCI output file.

    Returns:
        result (dict): Parsed record in a dict.
    """
    line = next(fp).strip()
    assert line[0] == "#", line
    rec_id = line.split(" ")[-1]

    line = next(fp).strip()  # "ANARCI numbered" or "//"
    if line == "//":
        return None

    try:
        rec_idx = rec_id.split("-")[-1]
        rec_idx = int(rec_idx) if rec_idx.isdecimal() else 0

        # Domain 1 of 1
        # Most significant HMM hit
        while not line.startswith("#|"):
            line = next(fp)

        # |species|chain_type|e-value|score|seqstart_index|seqend_index|
        keys = line.split("|")[1:-1]

        line = next(fp)
        assert line.startswith("#|"), line
        vals = line.split("|")[1:-1]

        meta = dict(zip(keys, vals))

        chain_type = meta["chain_type"]
        assert chain_type in (
            "H",
            "L",
            "A",
            "B",
            "K",
        ), f"{rec_id} chain type is {chain_type}"

        line = next(fp)  # Scheme = imgt

        imgt = {k: [] for k in IMGT_BOUNDARIES}
        line = next(fp).strip()
        while line != "//":
            chain, pos, *_, subpos, aa = line.split(" ")
            if aa != "-":
                ipos = int(pos)
                for k, (start, end) in IMGT_BOUNDARIES.items():
                    if start <= ipos <= end:
                        imgt[k].append((pos + subpos, aa))
                        break

            line = next(fp).strip()

            # Skip secondary domains
            if line.startswith("# Domain"):
                while line != "//":
                    line = next(fp).strip()
    except Exception as e:
        raise Exception(f"Failed parsing record {rec_id}.") from e

    return {
        "idx": rec_idx,
        "anarci_chain_type": chain_type,
        "anarci_species": meta["species"],
        "anarci_score": float(meta["score"]),
        "anarci_e_value": meta["e-value"],
        "anarci_seqstart_index": int(meta["seqstart_index"]),
        "anarci_seqend_index": int(meta["seqend_index"]),
        "imgt_fwr1": imgt["imgt_fwr1"],
        "imgt_cdr1": imgt["imgt_cdr1"],
        "imgt_fwr2": imgt["imgt_fwr2"],
        "imgt_cdr2": imgt["imgt_cdr2"],
        "imgt_fwr3": imgt["imgt_fwr3"],
        "imgt_cdr3": imgt["imgt_cdr3"],
        "imgt_fwr4": imgt["imgt_fwr4"],
    }


def file_parser(path: str):
    """
    Creates a parser iterator over the the ANRCI output file.
    Currently assumes human data and IMGT numbering.

    Params:
        path (str): Path to the ANRCI output file.
    """
    with open(path) as fp:
        try:
            while True:
                rec = parse_record(fp)
                if rec is not None:
                    yield rec
        except StopIteration:
            return


def execute_anarci(data: str, output_path: Path, scheme="imgt", ncpu=32):
    """
    Run ANARCI and return the annotations and a list of IMGT region
    AA assignments from ANARCI results.

    Params:
        sequence (str): Sequence or the path to FASTA file to annotate.
    Returns:
        Chain annotations dictionary.
    """
    ARG_CMD = [
        "ANARCI",
        "--scheme",
        scheme,
        "--ncpu",
        f"{ncpu}",
        "-i",
        data,
        "-o",
        output_path.as_posix(),
    ]
    result = subprocess.run(ARG_CMD, capture_output=True)
    result.output_file = output_path.as_posix()
    return result


def aa_regions(anarci):
    """
    Generates a list of IMGT region AA assignments from Anarci results dict.
    Parameters
    ----------
    anarci : dict
        Dictionary with results from an Anarci run.

    Returns
    -------
    dict
        A dictionary with imgt regions as keys and tuples of indices (start, end) as values.
    """
    regions = [
        f"imgt_{r}" for r in ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")
    ]
    start = anarci["anarci_seqstart_index"]
    limits = ct.reduce(
        lambda acc, x: acc + [acc[-1] + len(anarci[x])], regions, [start]
    )
    return dict(zip(regions, zip(limits, limits[1:])))


def annotate_chain(sequence: str):
    """
    Run ANARCI and parse the annotations.

    Params:
        sequence (str): Sequence or the path to FASTA file to annotate.
        entity_id (str): Entity id
    Returns:
        Chain annotations
    """
    if sequence is np.NaN or len(sequence) == 0:
        return {}
    sequence = sequence.replace(PROTEIN_X, PROTEIN_G)
    sequence = sequence.replace(PROTEIN_U, PROTEIN_G)

    with tempfile.NamedTemporaryFile(suffix=".anarci") as temp_file:
        results = execute_anarci(
            data=sequence,
            output_path=Path(temp_file.name),
        )
        if results.returncode != 0:
            raise Exception(results.stderr)
        out = list(file_parser(results.output_file))

    if not out:
        return {"anarci_chain_type": None}

    out[0]["aa_regions"] = aa_regions(out[0])
    start = out[0]["anarci_seqstart_index"]
    out[0]["variable"] = (start, start + out[0]["aa_regions"]["imgt_fwr1"][-1])

    return out[0]
