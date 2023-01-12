#You need to install the following python packages
#pytorch, vit_pytorch.
import argparse
import sys
import pickle
import os
import torch
from vit_pytorch import ViT
import time
import torch.nn.functional as F
import torch.optim as optim
import wandb
torch.manual_seed(97)

sys.path.append('./src/data')
from make_dataset import MNIST

wandb.init(entity="dtu_mlops_group24",project="dtu_mlops24")


def train():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('-b','--batch_size', default=128, type=int)
    parser.add_argument('-e','--epochs', default=5, type=int)
    parser.add_argument('-d','--device')
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    if args.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    print("Device: ", device)

    start_time = time.time()
    model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
    model.to(device)

    with open('./data/processed/dataset_train.pt', 'rb') as f:
    # Deserialize the object and recreate it in memory
        train_set = pickle.load(f)
    
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    optimz = optim.Adam(model.parameters(), lr=args.lr)


    model.train()
    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(dataloader):
            optimz.zero_grad()
            out = F.log_softmax(model(data.to(device)), dim=1)
            loss = F.nll_loss(out, target.to(device))
            loss.backward()
            optimz.step()
            
            if i % 1 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(len(train_set)) +
                    ' (' + '{:3.0f}'.format(100 * i / len(dataloader)) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                wandb.log({"train_loss":loss.item()})

    torch.save(model.state_dict(),'models/trained_model.pt')
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == '__main__':
    train()