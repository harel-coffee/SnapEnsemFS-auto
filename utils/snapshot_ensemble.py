import os 
import copy 
import time
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

import warnings
warnings.filterwarnings('ignore')


# hyperparameters
train_batch_size = 4
learning_rate_init = 0.0002
num_cycles = 5
num_epochs = 100
momentum=0.9
phases = ['training','validation']

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Cosine Annealing function (for cyclic LR Scheduler)
def cosine_annealing(epoch, epochs_per_cycle, lrate_max):
	cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
	return lrate_max/2 * (math.cos(cos_inner) + 1)


def train_model(model, criterion, optimizer, data_loader, num_epochs=50, num_cycles=5):

    # making variables global so as to access them
    global train_batch_size, learning_rate_init, phases, device, momentum

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    snapshots = []

    epochs_per_cycle = int(num_epochs/num_cycles)
    
    for cycle in range(num_cycles):

        print('================================')
        print(f'Snapshot [{cycle+1}/{num_cycles}]')
        print('================================')

        best_model_wts = copy.deepcopy(model.state_dict())
        best_accuracy = 0

        for epoch in range(epochs_per_cycle):

            # set learning rate for current epoch
            temp_lr = cosine_annealing(epoch, epochs_per_cycle, learning_rate_init)
            optimizer.state_dict()['param_groups'][0]['lr'] = temp_lr
            print(f'\nEpoch [{epoch+1}/{epochs_per_cycle}] | Learning rate = {temp_lr}\n')
            
            for phase in phases:
                if phase == 'training':
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0
                epoch_corrects = 0    
                
                for ii, (images,labels,_) in enumerate(data_loader[phase]):

                    images = images.to(device)
                    labels = labels.to(device)
                    
                    with torch.set_grad_enabled(phase == 'training'):
                        _, outputs = model(images)
                        
                        _,preds = torch.max(outputs,1) 
                        loss = criterion(outputs,labels)

                        # backward + optimize only if in training phase
                        if phase == 'training':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                    epoch_corrects += torch.sum(preds == labels.data)
                    epoch_loss += loss.item() * images.size(0)

                epoch_accuracy = epoch_corrects/len(data_loader[phase]) 
                epoch_loss /= len(data_loader[phase])

                # store statistics
                if phase == 'training':
                    epoch_loss = epoch_loss / train_batch_size
                    train_loss.append(epoch_loss)
                    epoch_accuracy = epoch_accuracy / train_batch_size
                    train_acc.append(epoch_accuracy)
                if phase == 'validation':
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_accuracy)

                print(f'Phase: {phase} | Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.6f}')

                # deep copy the best model weights
                if phase == 'validation' and epoch_accuracy >= best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'====> Best accuracy reached so far at Epoch {epoch+1} | Accuracy = {best_accuracy:.6f}')
            
            print('-------------------------------------------------------------------------')


        # training complete for current cycle
        print(f'>>> Saved model: Snapshot {cycle+1} | Best Validation Accuracy: {best_accuracy:4f}')
        model.load_state_dict(best_model_wts)
        snapshot = copy.deepcopy(model)
        snapshots.append(snapshot)

    
    history = {
        'train_loss' : train_loss.copy(),
        'train_acc' : train_acc.copy(),
        'val_loss' : val_loss.copy(),
        'val_acc' : val_acc.copy()
    }

    return snapshots, history


# plot training history curve
def plot_history(history):
    '''
    the function assumes history is a dictionary having four keys:
    1. train_loss	2. val_loss		3. train_acc	4. val_acc
    the function also assumes the values are on GPU
    '''
    num_epochs = len(history)

    # comment out the following lines (146-148) if all values are on CPU
    for i in range(num_epochs):
        history['train_acc'][i]=history['train_acc'][i].cpu().numpy().item()
        history['val_acc'][i]=history['val_acc'][i].cpu().numpy().item()

    fig, axes = plt.subplots(2,1,figsize=(8,8))
    fig.tight_layout(pad=5)

    iters = np.arange(num_epochs) + 1
    fig.suptitle('Model training curve')

    axes[0].set_title('Loss over epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='best')

    axes[0].plot(iters, history['train_loss'],label='Training Loss')
    axes[0].plot(iters, history['val_loss'],label='Validation Loss')

    axes[1].set_title('Accuracy over epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(loc='best')

    axes[1].plot(iters, history['train_acc'],label='Training Accuracy')
    axes[1].plot(iters, history['val_acc'],label='Validation Accuracy')

    plt.savefig('outputs/TL_history.jpg', dpi=300)
    fig.show()


# evaluating model and getting features of every image
def eval_model_extract_features(features, true_labels, img_paths, model, index, dataloader, phase):
      
    with torch.no_grad():
        # for entire dataset
        n_correct = 0
        n_samples = 0

        model.eval()

        for images,labels,paths in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            true_labels.append(labels)
            img_paths.append(paths)
            
            ftrs, outputs = model(images)
            features.append(ftrs)

            _,preds = torch.max(outputs,1)

            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
                
        accuracy = n_correct/float(n_samples)

        print(f'Accuracy of model {index+1} on {phase} set = {(100.0 * accuracy):.4f} %')

    return features, true_labels, img_paths


# get features as arrays
def get_features(features, true_labels, img_paths):
    ftrs = features.copy()
    lbls = true_labels.copy()
    paths = img_paths.copy()

    for i in range(len(ftrs)):
        ftrs[i] = ftrs[i].cpu().numpy()

    for i in range(len(lbls)):
        lbls[i] = lbls[i].cpu().numpy()

    filenames = []
    for i in range(len(paths)):
        for name in paths[i]:
            filenames.append(os.path.basename(name))

    # convert to numpy array
    ftrs = np.array(ftrs)
    lbls = np.array(lbls)

    print(f'shape of feature set: {ftrs.shape}')

    n_samples = ftrs.shape[0] * ftrs.shape[1]
    n_features = ftrs.shape[2]
    ftrs = ftrs.reshape(n_samples, n_features)

    n_lbls = lbls.shape[0]
    lbls = lbls.reshape(n_lbls)

    return ftrs, lbls, filenames