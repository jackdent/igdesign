from pathlib import Path
import numpy as np
from igdesign.data.pdb_parser import PDBParserAnnotator, PDBParserAnnotatorV2

#pdb_path = Path(__file__).parent / "5j13.pdb"
pdb_path_old = Path(__file__).parent / "data" / "input_old.pdb"
pdb_path_new = Path(__file__).parent / "data" / "input_new.pdb"

for chain_id in ["A", "B", "C"]:

    # load old pdb with old parser: missing residues are cut off from sequence
    # because igdesign reads missing residues from PDB remarks and we don't have these.
    parser = PDBParserAnnotator()
    data_v1 = parser.parse_PDB(pdb_path_old, chain = chain_id, backbone_only=True, skip_check=False)

    # load new pdb with new parser: missing residues are present
    # changes: 1) add correct seqres to pdb file, 2) parse missing residues from seqres instead of PDB remarks
    parser_v2 = PDBParserAnnotatorV2()
    data_v2 = parser_v2.parse_PDB(pdb_path_new, chain = chain_id, backbone_only=True, skip_check=False)

    if chain_id in ["B", "C"]:
        assert data_v1["sequence"] == data_v2["sequence"]
        assert data_v1["coords"].shape == data_v2["coords"].shape
        assert np.allclose(data_v1["coords"], data_v2["coords"], equal_nan = True)
    else:
        assert len(data_v1["sequence"]) == 109
        assert len(data_v2["sequence"]) == 124

        assert len(data_v2["coords"]) == len(data_v2["sequence"])
