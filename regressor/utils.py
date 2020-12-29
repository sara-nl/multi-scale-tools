import sys, getopt
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import os
import shutil
import random
from torch.utils import data
import torch

#create folder (used for saving weights)
def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.makedirs(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path)

def get_kind_of_path(input_path):

    #csv
    if ('.csv' in input_path):
        input_files = pd.read_csv(input_path, sep=',', header=None).values[:,0]
    #folder
    elif (os.path.isdir(input_path)):
        input_files = []
        list_files = os.listdir(input_path)
        
        for f in list_files:
            if ('.png' in f or '.jpg' in f):
                input_files.append(input_path+'/'+f)
    #file
    else:
        input_files = [input_path]

    return input_files

def change_magnification_name(array):
    new_array = []
    for a in array:
        
        if (a.is_integer()):
            new_array.append(int(a))
        else:
            new_array.append(a)
    
    return new_array

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

if __name__ == "__main__":
	pass