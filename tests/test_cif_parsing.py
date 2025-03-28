
# %%
from pathlib import Path
from igdesign.data import cif_parser

cif_path = Path(__file__).parent / "data" / "5j13_chaiinput.cif"
item = cif_parser.dataset_item_from_cif(cif_path.as_posix(), uid = "5j13", antigen_chain_id = "A", heavy_chain_id = "C", light_chain_id = "B")
print(item)
# %%
