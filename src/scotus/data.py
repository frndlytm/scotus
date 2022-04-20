from os import path

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

# from vibr import constants
from . import constants


# TODO:
#     TITLE: Data Loading and Preparation
#     AUTHOR: frndlytm
def load(data_dir: str = constants.DATA_DIR):
    inputs, outputs = (..., ...)
    train_idxs, val_idxs, test_idxs = (..., ..., ...)

    # RETURN DataLoader batch iterators on the data
    return {
        "d_in": inputs.shape[1],
        "d_out": outputs.shape[1],
        "datasets":{
            "train": TensorDataset(
                torch.from_numpy(inputs[train_idxs]),
                torch.from_numpy(outputs[train_idxs])
            ),
            "validate": TensorDataset(
                torch.from_numpy(inputs[val_idxs]),
                torch.from_numpy(outputs[val_idxs])
            ),
            "test": TensorDataset(
                torch.from_numpy(inputs[test_idxs]),
                torch.from_numpy(outputs[test_idxs])
            ),
        },
    }


