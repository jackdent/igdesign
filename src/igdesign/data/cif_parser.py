from typing import Any
from igdesign.tokenization import BACKBONE_ATOM_LIST
from chai.data.dataset.structure.load import download_structure, get_pdb_entities, structure_to_entities_data
from chai.data.io.clouds import temp_local_copy
from chai.data.parsing.antibodies import find_chains_and_cdrs
from chai.data.parsing.structure.all_atom_entity_data import AllAtomEntityData, load_structure
from chai.data.parsing.structure.entity_type import EntityType
import torch
from einops import rearrange



def dataset_item_from_cif(cif_path: str,
                          uid : str,
                          heavy_chain_id: str,
                          light_chain_id: str,
                          antigen_chain_id: str) -> dict[str, Any]:
    
    if cif_path.startswith("r2://"):
        entities = get_pdb_entities(pdb_id_or_path =cif_path)
    else:
        st = load_structure(cif_path)
        entities = structure_to_entities_data(st)
        
    return _dataset_item_from_entities(entities,
                                       uid,
                                       heavy_chain_id = heavy_chain_id,
                                       antigen_chain_id = antigen_chain_id,
                                       light_chain_id = light_chain_id)

def _dataset_item_from_entities(entities,
                                uid,
                                heavy_chain_id: str,
                                light_chain_id: str,
                                antigen_chain_id: str) -> dict:

    # filter to prot:
    entities = [e for e in entities if e.entity_type == EntityType.PROTEIN]
    
    # order chains: H, L, antigens
    metadata = find_chains_and_cdrs(entities)

    ordered_chain_ids = [heavy_chain_id, light_chain_id, antigen_chain_id]
    
    subchain2entity = {e.subchain_id : e for e in entities}
    ordered_entities = [subchain2entity[c] for c in ordered_chain_ids]

    # sequences:
    sequences = [e.sequence for e in ordered_entities]

    # coordinates:
    chains_coords = []
    for e in ordered_entities:
        chain_coo_and_masks = [e.get_coords_and_mask(atom) for atom in BACKBONE_ATOM_LIST]
        coords = torch.stack([x[0] for x in chain_coo_and_masks], dim = 0)
        mask =  torch.stack([x[1] for x in chain_coo_and_masks], dim = 0)
        coords[~mask] = torch.nan

        coords = rearrange(coords, "atoms n c -> n atoms c")
        chains_coords.append(coords)

    # annotations:
    chain_annotations : list[dict] = []
    for c in ordered_chain_ids:
        chain_data = metadata.get(c)
        assert isinstance(chain_data.regions, dict)
        if len(chain_data.regions) == 0 : # antigen
            chain_annotations.append({"epitope": None})
        else:
            annots_dense = chain_data.regions
            annots_range = {k: (min(v), max(v)+1) for k,v in annots_dense.items()}
            chain_annotations.append(annots_range)

    return dict(
        pdb = uid,
        chains = ordered_chain_ids,
        sequence = sequences,
        coords = chains_coords,
        annotations = chain_annotations,
        name = uid,
        num_samples = 1,
        sample_idx = None,
        heavy = heavy_chain_id,
        light = light_chain_id,
        antigens = antigen_chain_id,
    )