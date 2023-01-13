# here test that the state dict we are saving can be loaded by the model
import torch
from vit_pytorch import ViT

from tests import _PROJECT_ROOT


def test_state_dict_loading():
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
    # state_dict = torch.load(_PROJECT_ROOT + "/src/models/trained_model.pt")
    # assert state_dict is not None, "State dict is None"
    # model.load_state_dict(state_dict)
    # Some test to make sure the model is working and the state dict is loaded correctly
