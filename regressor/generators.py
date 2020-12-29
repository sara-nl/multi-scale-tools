import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
import os
import random
from torchvision import transforms
import albumentations as A
from PIL import Image

def find_next_magnification(mag, MAGNIFICATION):
    
    next_m = 0
    
    for a, m in enumerate(MAGNIFICATION):
        
        if (m>mag):
            
            next_m = m
    
    return next_m

def manipulate_resolution(img,scale, MAGNIFICATION):
    next_mag = find_next_magnification(scale, MAGNIFICATION)

    RATIO = scale/next_mag
    
    scaled_value = random.uniform(RATIO, 1)

    new_scale = scaled_value*scale/RATIO

    new_size = int(224*scale/new_scale)

    center_img = 112

    img_pil = Image.fromarray(img)
    cropped_img = img_pil.crop((center_img - new_size//2, center_img - new_size//2, center_img + new_size//2, center_img + new_size//2))
    
    new_image = cropped_img.resize((224,224))
    new_image = np.asarray(new_image)
    
    return new_image, new_scale

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

class Dataset_training(data.Dataset):

    def __init__(self, list_IDs, labels, mode, prob, magnifications):

        self.list_IDs = list_IDs
        self.labels = labels
        self.mode = mode
        self.prob = prob
        self.magnifications = magnifications
        self.pipeline_transform = get_augmentation()
        self.preprocess = get_preprocess()
        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = Image.open(ID)
        new_image = np.asarray(X)
        y = self.labels[index]
        #data augmentation
        #geometrical

            #augmentation + scale
        if (self.mode=='train'):
            new_image = self.pipeline_transform(image=new_image)['image']
        
            p_scale = random.random()

            if(p_scale>self.prob and y<self.magnifications[0] and y>self.magnifications[-1]):
                new_image, y = manipulate_resolution(new_image,y, self.magnifications)

        #data transformation
        input_tensor = self.preprocess(new_image).type(torch.FloatTensor)
        
        return input_tensor, np.asarray([y])

class Dataset_test(data.Dataset):

    def __init__(self, list_IDs):

        self.list_IDs = list_IDs
        self.preprocess = get_preprocess()
        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = Image.open(ID)
        X = np.asarray(X)
        #data augmentation
        #geometrical

        #data transformation
        input_tensor = self.preprocess(X).type(torch.FloatTensor)
                
        return input_tensor

if __name__ == "__main__":
	pass