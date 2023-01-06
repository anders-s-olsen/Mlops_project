# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

import torch
import os
import wget
import numpy as np

cwd=os.getcwd()
data_path='./data'

os.chdir(data_path+"/raw")

files=os.listdir()

if 'mnist.npz' not in files:
    wget.download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')

data = np.load('mnist.npz')

os.chdir("../processed")

for key in data.keys():
    if key in ['x_train', 'x_test']:
        images=data[key].astype(float)
        images_mean = np.mean(images, axis=(1,2))
        images_std = np.std(images, axis=(1,2))
        norm_images = (images - images_mean[:, np.newaxis, np.newaxis]) / images_std[:, np.newaxis, np.newaxis]
        torch.save(torch.from_numpy(norm_images),f"{key}.pt")
    else:
        labels=torch.from_numpy(data[key])
        torch.save(labels,f"{key}.pt")

os.chdir(cwd)
