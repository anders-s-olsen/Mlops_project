import torch
import argparse
from make_dataset import MNIST
from torchvision.utils import save_image
import pickle
import sys
sys.path.append("./src/data")


parser = argparse.ArgumentParser(description='Give an integer')
parser.add_argument('integer', type=int)
# add any additional argument that you want
args = parser.parse_args(sys.argv[1:])


with open("./data/processed/dataset_test.pt", "rb") as f:
    # Deserialize the object and recreate it in memory
    test_set = pickle.load(f)

example, _ = test_set[args.integer]


save_image(example, f'mnist_example{args.integer}.png')
