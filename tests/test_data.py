import os.path
import pickle
import sys

import __main__
import pytest
import torch

from src.data.make_dataset import MNIST

setattr(__main__, "MNIST", MNIST)


# @pytest.mark.skipif(
#     not os.path.exists(_PATH_DATA + "/processed/dataset_test.pt"),
#     reason="Data files not found",
# )
def test_data():
    with open("./data/processed/dataset_test.pt", "rb") as f:
        dataset_test = pickle.load(f)

    with open("./data/processed/dataset_train.pt", "rb") as f:
        dataset_train = pickle.load(f)

    assert isinstance(dataset_test, MNIST), "Test dataset is not of type MNIST"
    assert isinstance(dataset_train, MNIST), "Train dataset is not of type MNIST"

    assert dataset_test.data.shape == (
        10000,
        1,
        28,
        28,
    ), "Test data shape is wrong, should be (10000, 28, 28)"
    assert dataset_test.targets.shape == (
        10000,
    ), "Test targets shape is wrong, should be (10000,)"
    assert dataset_train.data.shape == (
        60000,
        1,
        28,
        28,
    ), "Train data shape is wrong, should be (60000, 28, 28)"
    assert dataset_train.targets.shape == (
        60000,
    ), "Train targets shape is wrong, should be (60000,)"
