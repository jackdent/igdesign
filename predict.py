import os
import argparse
import yaml
import pathlib
import pickle

import pandas as pd
import torch
from pytorch_lightning import seed_everything

from igdesign.model_wrapper import LMDesignIFWrapper
from igdesign.inference_utils import sample
from igdesign.utils import safe_to_device
from igdesign.data.datasets.pdb_antibody import PdbAntibodyDataset


def setup_configs(config_name=None):
    if config_name is None:
        parser = argparse.ArgumentParser(description="Process configuration name.")
        parser.add_argument("--config_name", type=str, default="1n8z.yaml")
        args = parser.parse_args()
        config_name = args.config_name

    current_dir = pathlib.Path(__file__).parent.resolve()
    configs_filepath = os.path.join(current_dir, "configs", config_name)
    with open(configs_filepath, "r") as f:
        cfg = yaml.safe_load(f)
    if (
        "lcdr1" in cfg["region_order"]
        or "lcdr2" in cfg["region_order"]
        or "lcdr3" in cfg["region_order"]
    ):
        cfg["condition_on_light_chain"] = False
        cfg["predict_light_chain"] = True

    cfg["batch_size"] = 1
    return cfg


def check_configs(cfg):
    """Run a bunch of assertions to verify the configs are legitimate before continuing."""
    for field in cfg["region_order"]:
        assert (
            field in cfg["regions"].keys()
        ), f'Design region {field} must be defined in the "regions" section'
    assert not (
        cfg["condition_on_light_chain"] and cfg["predict_light_chain"]
    ), "You cannot both predict the light chain and condition on it."
    assert os.path.exists(cfg["structure_path"]), "Structure path does not exist."
    assert os.path.exists(
        cfg["lmdesign_checkpoint"]
    ), "LM-Design model checkpoint path does not exist."
    assert os.path.exists(
        cfg["pmpnn_checkpoint"]
    ), "ProteinMPNN model checkpoint path does not exist."


def load_model(cfg):
    ckpt = cfg["lmdesign_checkpoint"]
    pmpnn_path = cfg["pmpnn_checkpoint"]
    model = LMDesignIFWrapper.load_from_checkpoint(
        ckpt, strict=False, pmpnn_path=pmpnn_path
    )
    model = model.eval()
    model.cuda()
    return model


def load_single_example(model, cfg):
    if cfg["condition_on_antigen"]:
        antigen_chain_id = cfg["antigen_chain_id"]

    pdb_path = cfg["structure_path"]
    pdb_name = pdb_path.split("/")[-1].split(".")[0]
    batch_size = cfg["batch_size"]

    if cfg["epitope_idxs_or_all"] == "all":
        epitope = {}  # dummy
    else:
        epitope = {antigen_chain_id: {"indices": cfg["epitope_idxs_or_all"]}}

    dataset_cfg = {
        "pdb": pdb_name,
        "pdb_path": pdb_path,
        "heavy": {
            "chain": cfg["heavy_chain_id"],
            "has_sequence": True,
            "has_coords": True,
            "sequence": None,
        },
        "light": {
            "chain": cfg["light_chain_id"],
            "has_sequence": True,
            "has_coords": True,
            "coords": None,
        },
        "antigens": [
            {
                "chain": cfg["antigen_chain_id"],
                "has_sequence": True,
                "has_coords": True,
                "sequence": None,
            }
        ],
        "epitopes": epitope,
        "num_samples": int(batch_size),
        "name": pdb_name,
    }
    if cfg["epitope_idxs_or_all"] == "all":
        dataset = PdbAntibodyDataset(
            [dataset_cfg], include_light_chains=True, ag_crop_method="none"
        )
    else:
        dataset = PdbAntibodyDataset([dataset_cfg], include_light_chains=True)
    batch = [dataset[idx] for idx in range(int(batch_size))]
    batch = safe_to_device(batch, model.device)
    collated_batch = model.collate(batch, precollate=dataset.collate)
    collated_batch["tokenized_sequences"] = collated_batch["tokenized_sequences"].long()
    return collated_batch


def run_inference(cfg=None, config_name=None, save_root=None):
    cfg = setup_configs(config_name) if cfg is None else cfg
    check_configs(cfg)

    model = load_model(cfg)
    seed_everything(cfg["random_seed"])
    batch = load_single_example(model, cfg)

    # Compute chain offsets
    light_chain_offset = 0
    light_chain_offset = (batch["chain_ids"][0] == 0).sum().item()

    # Convert AA positions to tensors and compute offset_positions, which are positions + chain offsets
    for region in cfg["regions"].keys():
        input_positions = cfg["regions"][region]["positions"]
        if isinstance(input_positions, str):
            input_positions = eval(input_positions)
        input_positions = torch.LongTensor(input_positions)
        cfg["regions"][region]["positions"] = input_positions

        if cfg["regions"][region]["chain"] == "heavy":
            cfg["regions"][region]["offset_positions"] = cfg["regions"][region][
                "positions"
            ]
        elif cfg["regions"][region]["chain"] == "light":
            cfg["regions"][region]["offset_positions"] = (
                cfg["regions"][region]["positions"] + light_chain_offset
            )

    save_path = cfg["save_path"]
    df = sample(model=model, batch=batch, cfg=cfg, save_root=save_root)

    # Parse CDRs from "sampled_seq" column
    df[cfg["region_order"]] = df.sampled_seq.apply(lambda x: pd.Series(x.split("|")))
    df = df.drop(columns=["sampled_seq"])
    df = df[
        cfg["region_order"]
        + [col for col in df.columns if col not in cfg["region_order"]]
    ]

    # Save results and config
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    cfg_save_path = save_path.replace(".csv", "_config.pkl")
    with open(cfg_save_path, "wb") as filehandle:
        pickle.dump(cfg, filehandle)
    print(f"Done generating and scoring sequences. Results saved to {save_path}")


if __name__ == "__main__":
    run_inference()
