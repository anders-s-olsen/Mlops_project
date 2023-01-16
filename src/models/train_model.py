#You need to install the following python packages
#pytorch, vit_pytorch.
import sys
import pickle
import os
import torch
from vit_pytorch import ViT
import time
import torch.nn.functional as F
import torch.optim as optim
import wandb
import hydra
import subprocess
from hydra.utils import get_original_cwd
from google.cloud import storage
torch.manual_seed(97)

sys.path.append('./src/data')
from make_dataset import MNIST
wandb.init(entity="dtu_mlops_group24",project="dtu_mlops24")

@hydra.main(version_base=None,config_path=".",config_name="config.yaml",)
def train(cfg):
    os.chdir(get_original_cwd())
    hps = cfg.hyperparameters
    print(hps)

    if hps.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = hps.device

    print("Device: ", device)

    start_time = time.time()
    model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
    model.to(device)

    with open('./data/processed/dataset_train.pt', 'rb') as f:
    # Deserialize the object and recreate it in memory
        train_set = pickle.load(f)
    with open('./data/processed/dataset_test.pt', 'rb') as f:
        test_set = pickle.load(f)
    
    
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=hps.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=hps.batch_size, shuffle=False)
    optimz = optim.Adam(model.parameters(), lr=hps.lr)


    
    for epoch in range(hps.epochs):
        for i, (data, target) in enumerate(dataloader):
            model.train()
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
            if i % 10 == 0:
                with torch.no_grad():
                    model.eval()
                    test_acc = 0
                    test_loss = 0
                    for i, (data, target) in enumerate(testloader):
                        out = F.log_softmax(model(data.to(device)), dim=1)
                        loss = F.nll_loss(out, target.to(device))
                        test_loss += loss.item()
                        topval,topclass = out.topk(1,dim=1)
                        equals = topclass==target.view(*topclass.shape)
                        testacc = torch.mean(equals.type(torch.Tensor))
                        test_acc+=testacc.item()
                    wandb.log({"test_loss":test_loss})
                    wandb.log({"test_accuracy":100*test_acc/len(testloader)})
            
    torch.save(model.state_dict(),'models/trained_model.pt')

    storage_client = storage.Client()
    bucket = storage_client.bucket('model_checkpoints_group24')
    blob = bucket.blob('trained_model.pt')
    blob.upload_from_filename('models/trained_model.pt')

    #bucket = storage.Client().bucket('gs://model_checkpoints_group24/')
    #blob = bucket.blob('/trained_model.pt')
    #blob.upload_from_filename('models/trained_model.pt')
    print(f"Saved model files in 'gs://model_checkpoints_group24/trained_model.pt")
    # subprocess.check_call([
    #     'gsutil', 'cp', 'models/trained_model.pt',
    #     'gs://model_checkpoints_group24'
    # ])

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == '__main__':
    train()