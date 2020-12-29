import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torch.utils.data
import os
import shutil


#create folder (used for saving weights)
def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.makedirs(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path)

def freeze_unfreeze(freeze, model):
    if ('resnet' in CNN_TO_USE):
        ELEM_TO_FREEZE = 8
    elif ('densenet' in CNN_TO_USE):
        ELEM_TO_FREEZE = 5
    if (freeze):
        
        if (torch.cuda.device_count()>1):

            for i in range(ELEM_TO_FREEZE):
                child = model.conv_layers.module[i]
                child.eval()
                for param in child.parameters():
                        param.requires_grad = False
        else:

            for i in range(ELEM_TO_FREEZE):
                child = model.conv_layers[i]
                child.eval()
                for param in child.parameters():
                        param.requires_grad = False

    else:

        if (torch.cuda.device_count()>1):

            for i in range(ELEM_TO_FREEZE):
                child = model.conv_layers.module[i]
                for param in child.parameters():
                        param.requires_grad = True
        else:

            for i in range(ELEM_TO_FREEZE):
                child = model.conv_layers[i]
                for param in child.parameters():
                        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    """
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    """
    return model

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))             if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)             if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx,1]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def change_magnification_name(array):
    new_array = []
    for a in array:
        
        if (a.is_integer()):
            new_array.append(int(a))
        else:
            new_array.append(a)
    
    return new_array

if __name__ == "__main__":
	pass