import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision
from vit_pytorch import ViT


def predict(img=None,model_state=None):
    device="cpu"
    if img == None:
        
        parser = argparse.ArgumentParser(description='Evaluating arguments')
        parser.add_argument('load_model_from', type=str)
        parser.add_argument('load_input_from', type=str)
        parser.add_argument('-d','--device')
        # add any additional argument that you want
        #args = parser.parse_args(sys.argv[1:])
        args, unknown = parser.parse_known_args()
        print(args)


        input = torchvision.io.read_image(args.load_input_from,torchvision.io.ImageReadMode.GRAY).float()
    else:
        input = torchvision.io.read_image(img,torchvision.io.ImageReadMode.GRAY).float()


    model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128)

    if model_state == None:
        model.load_state_dict(torch.load(args.load_model_from))
    else:
        model.load_state_dict(torch.load(model_state))

    model.to(device)
    model.eval()

    input = input.unsqueeze(0)

    # Normalize data
    mean = input.mean(dim=(1, 2, 3), keepdim=True)
    std = input.std(dim=(1, 2, 3), keepdim=True)

    input = (input - mean) / std

    with torch.no_grad():
        preds=F.softmax(model(input.to(device)), dim=1)
        pred=preds.argmax(dim=-1)
    
    if img == None: 
        print(f"The predicted label is: {pred.item()} with probability {preds.max().item():.2f}")
    else:
        return f"The predicted label is: {pred.item()} with probability {preds.max().item():.2f}"


if __name__ == '__main__':
    predict()
