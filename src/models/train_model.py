#You need to install the following python packages
#pytorch, vit_pytorch.
import torch
from vit_pytorch import ViT
import time
import torch.nn.functional as F
import torch.optim as optim
import wandb
import make_dataset_transforms

make_dataset_transforms.rundata()
wandb.init(entity="dtu_mlops_group24",project="dtu_mlops24")
lr = 0.003

torch.manual_seed(97)

def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    
    for i, (data, target) in enumerate(data_load):
        optimz.zero_grad()
        out = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()
        
        if i % 1 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())
            wandb.log({"train_loss":loss.item()})

def evaluate(model, data_load, loss_val):
    model.eval()
    
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            tloss += loss.item()
            csamp += pred.eq(target).sum()
        wandb.log({"validation_loss":tloss.item()})

    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

### Load data
#tr_data = torch.unsqueeze(torch.load("data/processed/x_train.pt"),dim=1)
#tr_labels = torch.load("data/processed/y_train.pt")
#tr_data_all = torch.utils.data.TensorDataset(tr_data,tr_labels)
#tr_load = torch.utils.data.DataLoader(tr_data_all,batch_size=64,shuffle=True)
#ts_data = torch.unsqueeze(torch.load("data/processed/x_test.pt"),dim=1)
#ts_labels = torch.load("data/processed/y_test.pt")
#ts_data_all = torch.utils.data.TensorDataset(ts_data,ts_labels)
#ts_load = torch.utils.data.DataLoader(ts_data_all,batch_size=64,shuffle=False)

### Load data
tr_data = torch.load("data/processed/train.pt")
tr_load = torch.utils.data.DataLoader(tr_data,batch_size=64,shuffle=True)
ts_data = torch.load("data/processed/x_test.pt")
ts_load = torch.utils.data.DataLoader(ts_data,batch_size=64,shuffle=False)

N_EPOCHS = 25

start_time = time.time()
model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
optimz = optim.Adam(model.parameters(), lr=lr)

trloss_val, tsloss_val = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_iter(model, optimz, tr_load, trloss_val)
    evaluate(model, ts_load, tsloss_val)
torch.save(model.state_dict(),'models/trained_model.pt')
print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')