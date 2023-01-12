import os.path

import pytest
import torch

from tests import _PATH_DATA


@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA + "/processed/x_train.pt"),
    reason="Data files not found",
)
def test_data():
    x_train = torch.load(_PATH_DATA + "/processed/x_train.pt")
    x_test = torch.load(_PATH_DATA + "/processed//x_test.pt")
    y_train = torch.load(_PATH_DATA + "/processed//y_train.pt")
    y_test = torch.load(_PATH_DATA + "/processed//y_test.pt")
    assert x_train.shape == (60000, 28, 28), "x_train shape is not (60000, 28, 28)"
    assert x_test.shape == (10000, 28, 28), "x_test shape is not (10000, 28, 28)"
    assert y_train.shape == (60000,), "y_train shape is not (60000,)"
    assert y_test.shape == (10000,), "y_test shape is not (10000,)"
