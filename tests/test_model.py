import pytest
import torch
from vit_pytorch import ViT


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (torch.rand(128, 1, 28, 28), (128, 10)),
        (torch.rand(64, 1, 28, 28), (64, 10)),
        (torch.rand(32, 1, 28, 28), (32, 10)),
    ],
)
def test_model(test_input, expected):
    model = ViT(
        image_size=28,
        patch_size=4,
        num_classes=10,
        channels=1,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
    )
    assert model is not None, "Model is None"
    assert model(test_input).shape == expected, "Model output shape is not (64, 10)"
