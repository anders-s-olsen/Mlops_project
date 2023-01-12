import torch
from torch.utils.data import Dataset
import os
import wget
import numpy as np
import pickle

class MNIST(Dataset):
    def __init__(self,train):
        self.download_data()
        content = np.load("./data/raw/mnist.npz",allow_pickle=True)
        if train:
            data = torch.tensor(content['x_train'].astype(float)).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['y_train'])
        else:
            data = torch.tensor(content['x_test'].astype(float)).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['y_test'])

        # Normalize data
        mean = data.mean(dim=(1,2,3), keepdim=True)
        std = data.std(dim=(1,2,3), keepdim=True)

        data = (data - mean) / std

        self.data = data
        self.targets = targets


    def download_data(self):
        files = os.listdir('./data/raw')
        if "mnist.npz" not in files:
            wget.download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', out = './data/raw')

    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]

if __name__ == "__main__":
    dataset_train = MNIST(train=True)
    dataset_test = MNIST(train=False)
    
    # Open a file for writing
    with open('./data/processed/dataset_train.pt', 'wb') as f:
    # Serialize the object and write it to the file
        pickle.dump(dataset_train, f)

    # Open a file for writing
    with open('./data/processed/dataset_test.pt', 'wb') as f:
    # Serialize the object and write it to the file
        pickle.dump(dataset_test, f)

