import os
import yaml
import argparse

import torch
import numpy as np


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


def train_val_split(dset_len, train_size, val_size, seed):
    assert (train_size is None) + (val_size is None) <= 1, (
        "Only one of train_size, val_size is allowed to be None."
    )

    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size

    if train_size is None:
        train_size = dset_len - val_size
    elif val_size is None:
        val_size = dset_len - train_size

    if train_size + val_size  > dset_len:
        if is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) splits ended up with a negative size."
    )

    total = train_size + val_size
    
    assert (
            dset_len >= total
    ), f"The dataset ({dset_len}) is smaller than the combined split sizes ({total})."

    if total < dset_len:
        print(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int32)
    idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size: train_size + val_size]

    return np.array(idx_train), np.array(idx_val)


def number(text):
    r"""
    Converts a string to a number.
    """
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int

    return num_float


def make_splits(
    dataset_len: int,
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
    filename: str = None,
    splits=None,
):
    r"""
    Creates train, validation and test splits for a dataset.

    Parameters
    ----------
    dataset_len : int
        The length of the dataset.
    train_size : float, optional
        The size of the training split in percentage or absolute number. The default is 0.8.
    val_size : float, optional
        The size of the validation split in percentage or absolute number. The default is 0.1.
    seed : int, optional
        The seed for the random number generator. The default is 42.
    filename : str, optional
        The filename to save the splits to. The default is None.
    splits : str, optional
        The filename of the splits to load. The default is None.

    Returns
    -------
    tuple of torch.Tensor
        The indices of the training, validation and testing splits.
    """
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train, idx_val = train_val_split(
            dataset_len, train_size, val_size, seed
        )

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(
            filename, idx_train=idx_train, idx_val=idx_val
        )

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def save_argparse(
        args: argparse.Namespace,
        filename: str,
        exclude: list = None
):
    r"""
    Saves the argparse namespace to a file.

    Parameters
    ----------
    args : argparse.Namespace
        The argparse namespace to save.
    filename : str
        The filename to save the argparse namespace to.
    exclude : list, optional
        A list of keys to exclude from the argparse namespace. The default is None.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


