import os
import pandas as pd
import numpy as np
import torch

IMAGES_DATA_DIR = '../data/images/'
MASKS_DATA_DIR = '../data/masks/'

x_train_dir = os.path.join(IMAGES_DATA_DIR, 'train')
y_train_dir = os.path.join(MASKS_DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(IMAGES_DATA_DIR, 'val')
y_valid_dir = os.path.join(MASKS_DATA_DIR, 'val_labels')

x_test_dir = os.path.join(IMAGES_DATA_DIR, 'test')
y_test_dir = os.path.join(MASKS_DATA_DIR, 'test_labels')

TRAINING = True

# Set num of epochs
EPOCHS = 12

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
