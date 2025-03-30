
# %%
from omegaconf import OmegaConf
import torch
import pandas as pd
from chai.third_party.igdesign.igdesign_distributed import get_igdesign_model, load_single_example, run_inference
from igdesign import inference_utils  # type: ignore
import warnings

def get_sequence_perplexity(sequence_regions : dict[str, str], cfg : dict, model,) -> pd.DataFrame:
    """
    score CDR regions for a given input pdb and config
    """
    #TODO: set seed.

    assert sequence_regions.keys() == set(cfg["region_order"])
    assert cfg["batch_size"] == 1

    batch = load_single_example(model, cfg)
    light_chain_offset = (batch["chain_ids"][0] == 0).sum().item()

    seqs = batch["tokenized_sequences"].clone()

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

    # update sequence tensor with input CDR regions
    for region in cfg["region_order"]:
        start = cfg["regions"][region]["offset_positions"][0].item()
        end = cfg["regions"][region]["offset_positions"][-1].item() + 1
        encoded = inference_utils.aa_to_tensor(sequence_regions[region])[:,0] # (n,1 -> n)
        seqs[:, start:end] = encoded

    with torch.no_grad():
        all_losses = inference_utils.compute_independent_loss(cfg, seqs, batch, model)

    # copied code: post-process losses into per-region values, convert to df
    all_losses = {key: torch.stack(value) for key, value in all_losses.items()}
    combined_seqs = list()
    for seq_idx, seq in enumerate(seqs):
        combined_str = ""
        for region_idx, region_name in enumerate(cfg["region_order"]):
            region_subset = (
                seq[cfg["regions"][region_name]["offset_positions"]].cpu().numpy()
            )
            combined_str += inference_utils.tensor_to_aa(region_subset)
            if region_idx != len(cfg["region_order"]) - 1:
                combined_str += "|"  # region delimiter
        combined_seqs.append(combined_str)

    df = pd.DataFrame(combined_seqs, columns=["sampled_seq"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for loss_type in all_losses.keys():  # all_losses[loss_type]: n_seqs x n_losses
            column_name = [f"ce_loss_{loss_type}"]
            all_loss_values = list()
            for seq_idx in range(seqs.shape[0]):
                loss_values = all_losses[loss_type][seq_idx].cpu().numpy().tolist()
                all_loss_values.append(
                    "|".join([str(round(value, 4)) for value in loss_values])
                )
            df.loc[:, column_name] = all_loss_values

    # Parse CDRs from "sampled_seq" column
    df[cfg["region_order"]] = df.sampled_seq.apply(lambda x: pd.Series(x.split("|")))
    df = df.drop(columns=["sampled_seq"])
    df = df[
        cfg["region_order"]
        + [col for col in df.columns if col not in cfg["region_order"]]
    ]

    return df


# %%
cfg = OmegaConf.to_container(OmegaConf.load("/workspaces/models/submodules/igdesign/test_data/5J13/igdesign_config.yaml"))
model = get_igdesign_model().to("cuda").eval()

# %%
df_out, _, _ = run_inference(model, cfg)
assert len(df_out) == 1

row = df_out.iloc[0]
sequence_regions_to_score = {k: row[k] for k in cfg["region_order"]}
expected_losses = {k: row[f"ce_loss_independent_{k}"] for k in cfg["region_order"]}

print(sequence_regions_to_score)
print(expected_losses)

# one example
# sequence_regions_to_score = {"hcdr1": "GFTFSTYA",
#                              "hcdr2": "IWYDGSNK",
#                              "hcdr3": "ARAPRYDWLYGAFDI"}
# expected_losses = {"ce_loss_independent_hcdr1": 0.1658,
#                    "ce_loss_independent_hcdr2": 0.2256,
#                    "ce_loss_independent_hcdr3": 0.8703}

# %%
df = get_sequence_perplexity(
    sequence_regions=sequence_regions_to_score,
    cfg=cfg,
    model=model,
)
assert len(df) == 1
results = df.iloc[0].to_dict()

for key, expected_sequence in sequence_regions_to_score.items():
    assert results[key] == expected_sequence

for key, expected_loss in expected_losses.items():
    loss_key = f"ce_loss_independent_{key}"
    print(key, results[loss_key], f"expected_loss: {expected_loss}")

# %%