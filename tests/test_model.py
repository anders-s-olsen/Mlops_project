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


# add this code to the file src/models/model.py
# # src/models/model.py
# def forward(self, x: Tensor):
#    if x.ndim != 4:
#       raise ValueError('Expected input to a 4D tensor')
#    if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
#       raise ValueError('Expected each sample to have shape [1, 28, 28]')


def test_error_on_wrong_shape():
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
    assert model is not None
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
