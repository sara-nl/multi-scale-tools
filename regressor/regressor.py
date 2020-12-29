import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import staintools
import torch.utils.data
from sklearn import metrics 
import os
import shutil
import argparse 
import random
import utils
import model


def regression(model,data,BATCH_SIZE):
    # Parameters bag
    batch_size_instance = BATCH_SIZE
    num_workers = 4

    params_test = {'batch_size': batch_size_instance,
            'shuffle': True,
            'num_workers': num_workers}

    import generators
    testing_set = generators.Dataset_test(data)
    testing_generator_strong = torch.utils.data.DataLoader(testing_set, **params_test)

    print("testing dataset")

    #read data
    tot_batches = int(len(data)/batch_size_instance)
    y_pred = []


    with torch.no_grad():
        for inputs in testing_generator_strong:
            inputs = inputs.to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            #accumulate values
            outputs_np = outputs.cpu().data.numpy()
                
            y_pred = np.append(y_pred,outputs_np)  

    return outputs_np.flatten()
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #parser parameters
    parser = argparse.ArgumentParser(description='Configurations to train models.')
    parser.add_argument('-o', '--model_path', help='model path',type=str)
    parser.add_argument('-i', '--input_folder', help='folder where the input data (csv files) are stored',type=str)
    parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)

    args = parser.parse_args()

    BATCH_SIZE = args.BATCH_SIZE
    BATCH_SIZE_str = str(BATCH_SIZE)

    models_path = args.model_path
    #models_path = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/regressor/magnification_'+MAGNIFICATIONS_str+'x/'+CNN_TO_USE+'/'
    utils.create_dir(models_path)
    #path model file
    model_weights_filename = models_path+'regressor.pt'

    model = torch.load(model_weights_filename)
    model.eval()
    model.to(device)

    #filenames data
    input_data_folder = args.input_folder
    #input_data_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/Regressor_partitions/'

    data = utils.get_kind_of_path(input_data_folder)

    magnifications = regression(model,data,BATCH_SIZE)

    print(magnifications)