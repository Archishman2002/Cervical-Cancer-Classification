!pip install torchmetrics timm
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
import seaborn as sns  
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, models, transforms
import torch
from matplotlib import pyplot as plt
import os
from cv2 import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torchmetrics 
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW,Adam # optmizers
import time
from tqdm import tqdm

%config Completer.use_jedi = False

data_path = "../input/224-224-cervical-cancer-screening/kaggle/train/train"
images  =  [glob.glob(os.path.join(data_path, d, "*.*")) for d in os.listdir(data_path)]
train_paths = np.hstack(images)
# Additional data
extra_1 = "../input/224-224-cervical-cancer-screening/kaggle/additional_Type_1_v2"
extra_2 = "../input/224-224-cervical-cancer-screening/kaggle/additional_Type_2_v2"
extra_3 = "../input/224-224-cervical-cancer-screening/kaggle/additional_Type_3_v2"
images1  =  [glob.glob(os.path.join(extra_1, d, "*.*")) for d in os.listdir(extra_1)]
images2  =  [glob.glob(os.path.join(extra_2, d, "*.*")) for d in os.listdir(extra_2)]
images3  =  [glob.glob(os.path.join(extra_3, d, "*.*")) for d in os.listdir(extra_3)]
train_paths = np.append(train_paths, np.hstack(images1))
train_paths = np.append(train_paths, np.hstack(images2))
train_paths = np.append(train_paths, np.hstack(images3))

print(f'In this train set we have got a total of {len(train_paths)}')
N_EPOCHS = 5
OUTPUT_PATH = './'
BATCH_SIZE = 10
# detect and define device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
device = torch.device(device)
cpu = torch.device('cpu')
