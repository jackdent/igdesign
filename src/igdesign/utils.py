import cytoolz as ct
import torch


def safe_to_device(x, device):
    """Places tensor objects in x on specified device"""
    if isinstance(x, dict):
        return {k: safe_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [safe_to_device(v, device) for v in x]
    # Base case
    return x.to(device) if torch.is_tensor(x) else x


@ct.curry
def resolve_mask(key, mask_dict, subset=None):
    if subset is not None:
        mask_dict = {key: val[subset] for key, val in mask_dict.items()}
    if isinstance(key, MutableSequence) or isinstance(key, tuple):
        return ct.reduce(torch.logical_or, map(resolve_mask(mask_dict=mask_dict), key))
    elif isinstance(key, str):
        if key in mask_dict:
            return mask_dict[key]
        return eval(key, mask_dict)
    elif isinstance(key, torch.Tensor):
        return key
    else:
        raise Exception(f"Cannot resolve {key}.")


@ct.curry
def resolve_pair_mask(pair, mask_dict, subset=None):
    if subset is not None:
        mask_dict = {key: val[subset] for key, val in mask_dict.items()}
    if pair is None:
        return pairwise_mask(
            torch.zeros_like(mask_dict["all"]), torch.zeros_like(mask_dict["all"])
        )
    if isinstance(pair, MutableMapping):
        return pairwise_mask(
            resolve_mask(pair["entity_a"], mask_dict),
            resolve_mask(pair["entity_b"], mask_dict),
        )
    elif isinstance(pair, MutableSequence):
        return ct.reduce(
            torch.logical_or, map(resolve_pair_mask(mask_dict=mask_dict), pair)
        )
    else:
        raise Exception(f"Cannot resolve {pair}.")
