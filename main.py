import os
import copy
import time
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, classification_report, plot_confusion_matrix, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

import utils
from utils.snapshot_ensemble import *
from utils.feature_selection import *
from utils.feature_ensemble import *
from utils.solution import *
from utils.dataset import *

from model import *
from PSO import * 

import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = './', help='Directory where the image data is stored')
parser.add_argument('--epochs', type=int, default = 100, help='Number of Epochs of training')
parser.add_argument('--batch_size', type=int, default = 4, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default = 0.0002, help='Learning Rate')
parser.add_argument('--momentum', type=float, default = 0.9, help='Momentum')
parser.add_argument('--num_clycles', type=int, default = 5, help='Number of cycles')
args = parser.parse_args()

########### snapshot ensembling phase ###########

# directory paths
DIR_PATH = args.data
if DIR_PATH[-1]=='/':
    DIR_PATH = DIR_PATH[:-1]
TRAIN_DIR_PATH = os.path.join(DIR_PATH,'train')
VAL_DIR_PATH = os.path.join(DIR_PATH,'val')


# image transformations
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transformations = {    
    'train' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=(-180,180), translate=(0.1,0.1), scale=(0.9,1.1), shear=(-5,5)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}


# getting the datasets
train_dataset = ImageFolderWithPaths(TRAIN_DIR_PATH,transform=transformations['train'])
val_dataset = ImageFolderWithPaths(VAL_DIR_PATH,transform=transformations['val'])
classes_to_idx = train_dataset.class_to_idx

print(f'Length of training dataset: {len(train_dataset)}')
print(f'Length of validation dataset: {len(val_dataset)}')
print(f'Classes in the dataset: {classes_to_idx}')


# hyperparameters
train_batch_size = args.batch_size
learning_rate_init = args.learning_rate
num_cycles = args.num_cycles
num_classes = len(classes_to_idx)
num_epochs = args.epochs
momentum = args.momentum
phases = ['training','validation']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ' + str(device))


# dataloaders
data_loader = {
    'training' : DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=4),
    'validation' : DataLoader(dataset=val_dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=4)
}

for phase in phases:
    print(f'Length of {phase} loader = {len(data_loader[phase])}')


# print data items from trainng dataloader
examples = iter(data_loader['training'])
images, labels, paths = examples.next()
print(f'Image shape: {images.shape} | Label shape: {labels.shape}') # batch_size=4
for path in paths:
    print(os.path.basename(path))
print('---------------------------------------')


# defining model, loss function, optimizer
model = get_model(device, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_init, momentum=momentum)


# print and test model
print("Model architecture:")
print(model)
test_model(model, device)
print('--------------------------------------')


# train model
start = time.time()
snapshots, history = train_model(model, criterion, optimizer, data_loader, num_epochs, num_cycles)    
duration = time.time() - start
print(f'Training complete in {(duration // 60):.0f}mins {(duration % 60):.0f}s')

# plot training curves
plot_history(history)

print('----------------------------------------------------\n\n')

# extract features using model snapshot and save into csv
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)
data_loader['training'] = train_loader

for idx, snapshot in enumerate(snapshots):
    features, true_labels, img_paths = [], [], []
    for phase in phases:
        features, true_labels, img_paths = eval_model_extract_features(features, true_labels, img_paths, snapshot, idx, data_loader[phase], phase)
    
    # convert tensors to numpy arrays
    features, true_labels, img_paths = get_features(features,true_labels, img_paths)    
    # print(len(features),len(true_labels),len(img_paths))
    
    # save to csv
    ftrs_df = pd.DataFrame(features)
    ftrs_df['label'] = true_labels.copy()
    ftrs_df['filename'] = img_paths.copy()
    ftrs_df.to_csv('outputs/snapshot_'+ str(idx+1) + '.csv',index=False)
    print(f'feature set for model snapshot {idx+1} saved successfully !')

print('-----------------------------------------------------------\n\n')


########### feature selection process ###########

# feature set paths
SNAPSHOT_1 = 'outputs/snapshot_1.csv'
SNAPSHOT_2 = 'outputs/snapshot_2.csv'
SNAPSHOT_3 = 'outputs/snapshot_3.csv'
SNAPSHOT_4 = 'outputs/snapshot_4.csv'
SNAPSHOT_5 = 'outputs/snapshot_5.csv'

# get concatenated feature set
df_concat = get_feature_set()
print("Concatenated feature set sample:")
print(df_concat.head())


# shuffle train and test separately
train_size = int(len(df_concat) * 0.8)
train_df = df_concat[:train_size].copy()
test_df = df_concat[train_size:1+len(df_concat)].copy()

train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)

df_concat = pd.concat([train_df, test_df], axis=0)


# get data and labels (X and y)
X = df_concat.iloc[:,0:(df_concat.shape[1]-2)]
y = df_concat['label']

X = np.array(X)
y = np.array(y)

print(f'Feature set dimensions: {X.shape} \nNo. of labels: {y.shape[0]}')

print('----------------------------------------')

# perform Feature Selection on the feature set
soln_PSO, conv_gph_PSO = PSO(num_agents=40, max_iter=40, data=X, label=y) 

# validate the feature selection algorithm
agent = soln_PSO.best_agent.copy()
cols = np.flatnonzero(agent)
validate_FS(X, y, agent, 'knn')

print('------------------------------------------------------')

