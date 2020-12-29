import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import torch.nn.functional as F
import torch.utils.data
import os
from torchvision import transforms

def get_preprocess():
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

def get_augmentation():
    prob = 0.5
    pipeline_transform = A.Compose([
        A.VerticalFlip(p=prob),
        A.HorizontalFlip(p=prob),
        A.RandomRotate90(p=prob),
        #A.ElasticTransform(alpha=0.1,p=prob),
        A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-30,20),val_shift_limit=(-15,15),p=prob),
        ])
    return pipeline_transform

class Dataset_multi_scale(data.Dataset):

    def __init__(self, list_IDs, MAGNIFICATION, mode_training, upper_level):

        self.N_MAGNIFICATIONS = len(MAGNIFICATION)
        self.pipeline_transform = get_augmentation()
        self.preprocess = get_preprocess()
        self.mode_training = mode_training
        self.upper_level = upper_level

        if (self.upper_level=='upper_region'):

            self.labels = list_IDs[:,-self.N_MAGNIFICATIONS:]
            self.list_IDs = list_IDs[:,:-self.N_MAGNIFICATIONS]

        elif (self.upper_level=='multi_center'):

            self.labels = list_IDs[:,-1:].flatten()
            self.list_IDs = list_IDs[:,:-1]

        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        X = []

        if (self.upper_level=='upper_region'):

            Y = []

            for i in range(self.N_MAGNIFICATIONS):

                # Select sample
                ID = self.list_IDs[index,i]
                # Load data and get label
                x = Image.open(ID)
                x = np.asarray(x)
                y = self.labels[index,i]
                #data augmentation
                #geometrical
                if (self.mode_training=='train'):
                    x = self.pipeline_transform(image=x)['image']
                input_tensor = self.preprocess(x).type(torch.FloatTensor)

                X.append(input_tensor)
                Y.append(np.asarray(y))
            #data transformation

        elif (self.upper_level=='multi_center'):

            Y = self.labels[index]
            Y = np.asarray(Y)

            for i in range(self.N_MAGNIFICATIONS):

                # Select sample
                ID = self.list_IDs[index,i]
                # Load data and get label
                x = Image.open(ID)
                x = np.asarray(x)
                
                #data augmentation
                #geometrical
                if (self.mode_training=='train'):
                    x = self.pipeline_transform(image=x)['image']
                input_tensor = self.preprocess(x).type(torch.FloatTensor)

                X.append(input_tensor)

            #data transformation

        return X, Y

class Dataset_single_scale(data.Dataset):

    def __init__(self, list_IDs, labels, mode_training):

        self.labels = labels
        self.list_IDs = list_IDs
        self.mode_training = mode_training
        self.pipeline_transform = get_augmentation()
        self.preprocess = get_preprocess()
        
        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        x = Image.open(ID)
        x = np.asarray(x)
        y = self.labels[index]
        #data augmentation
        #geometrical
        if (self.mode_training=='train'):
            x = self.pipeline_transform(image=x)['image']
        #data transformation
        input_tensor = self.preprocess(x).type(torch.FloatTensor)

        return input_tensor, np.asarray(y)

if __name__ == "__main__":
	pass