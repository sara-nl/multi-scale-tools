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

import torch.optim as optim

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-o', '--model_path', help='model path',type=str)
parser.add_argument('-i', '--input_folder', help='folder where the input data (csv files) are stored',type=str)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-f', '--file', help='filename where data are included',type=str, default='test')
parser.add_argument('-v','--verbose', help='verbose', type=bool, default=False)

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)

FILE = args.file

verbose = args.verbose 

seed = int(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))

#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")

models_path = args.model_path
#models_path = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/regressor/magnification_'+MAGNIFICATIONS_str+'x/'+CNN_TO_USE+'/'
utils.create_dir(models_path)
#path model file
model_weights_filename = models_path+'regressor.pt'

checkpoint_path = models_path+'checkpoints/'
utils.create_dir(checkpoint_path)

#CSV LOADING
print("CSV LOADING ")
#filenames data
input_data_folder = args.input_folder
input_data_folder = input_data_folder+'/'
#input_data_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/Regressor_partitions/'

csv_filename_testing = input_data_folder+FILE+'.csv'

#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values#[:2000]

MAGNIFICATION, _ = np.unique(test_dataset[:,1], return_counts=True)

print("CSV LOADED ")

print("MODEL DEFINITION")
#MODEL DEFINITION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

print("SETTING PARAMETERS")
# Parameters bag
batch_size_instance = BATCH_SIZE
num_workers = 4

params_test = {'batch_size': batch_size_instance,
          'shuffle': True,
          'num_workers': num_workers}

print("CREATING GENERATOR")
import generators
testing_set = generators.Dataset_training(test_dataset[:,0], test_dataset[:,1], 'test', 0.5, None)
testing_generator_strong = data.DataLoader(testing_set, **params_test)

print("testing dataset")

#read data
tot_batches = int(len(test_dataset)/batch_size_instance)
y_pred = []
y_true = []

model = torch.load(model_weights_filename)
model.eval()

with torch.no_grad():
    i = 0
    for inputs,labels in testing_generator_strong:
        inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
        
        # forward + backward + optimize
        outputs = model(inputs)

        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels.cpu().data.numpy()
               
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)

        if (i%100==0 and verbose==True):
            print('%d / %d' % (i, tot_batches))

            print("r2_score " + str(metrics.r2_score(y_true, y_pred)))
            print("explained_variance_score " + str(metrics.explained_variance_score(y_true, y_pred)))

    
        i = i+1

max_error_general_filename = checkpoint_path+'max_error_'+FILE+'.csv'
explained_variance_score_filename = checkpoint_path+'explained_variance_score_'+FILE+'.csv'
mean_squared_error_filename = checkpoint_path+'mean_squared_error_'+FILE+'.csv'
r2_score_filename = checkpoint_path+'r2_score_'+FILE+'.csv'

#k-score
try:
    max_error = metrics.max_error(y_true, y_pred)

    print("max_error " + str(max_error))

    max_errors = [max_error]

    File = {'val':max_errors}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(max_error_general_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    explained_variance_score = metrics.explained_variance_score(y_true, y_pred)

    print("explained_variance_score " + str(explained_variance_score))

    explained_variance_scores = [explained_variance_score]

    File = {'val':explained_variance_scores}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(explained_variance_score_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)

    print("mean_squared_error " + str(mean_squared_error))

    mean_squared_errors = [mean_squared_error]

    File = {'val':mean_squared_errors}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(mean_squared_error_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    r2_score = metrics.r2_score(y_true, y_pred)

    print("r2_score " + str(r2_score))

    r2_scores = [r2_score]

    File = {'val':r2_scores}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(r2_score_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

def normalize_value(val):
    
    dist_min = 1000000.0
    i = 0
    for a, m in enumerate(MAGNIFICATION):
        
        dist = abs(m-val)

        if (dist<=dist_min):
            dist_min = dist
            i = a
    return i

y_true_norm = []
for v in y_true:
    y_true_norm.append(normalize_value(v))

y_pred_norm = []
for v in y_pred:
    y_pred_norm.append(normalize_value(v))

y_true_norm = np.array(y_true_norm)
y_pred_norm = np.array(y_pred_norm)

kappa_score_general_filename = checkpoint_path+'kappa_score_'+FILE+'_norm.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_'+FILE+'_norm.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_'+FILE+'_norm.csv'
f1_macro_score_filename = checkpoint_path+'f1_macro_score_'+FILE+'_norm.csv'
f1_micro_score_filename = checkpoint_path+'f1_micro_score_'+FILE+'_norm.csv'

#k-score
try:
    k_score = metrics.cohen_kappa_score(y_true_norm,y_pred_norm, weights='quadratic')
    print("k_score " + str(k_score))

    kappas = [k_score]

    File = {'val':kappas}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(kappa_score_general_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    #confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true=y_true_norm, y_pred=y_pred_norm)
    print("confusion_matrix ")
    print(str(confusion_matrix))


    conf_matr = [confusion_matrix]
    File = {'val':conf_matr}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(confusion_matrix_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    acc_balanced = metrics.balanced_accuracy_score(y_true_norm, y_pred_norm, sample_weight=None, adjusted=False)
    print("acc_balanced " + str(acc_balanced))

    acc_balancs = [acc_balanced]

    File = {'val':acc_balancs}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(acc_balanced_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    f1_score = metrics.f1_score(y_true_norm, y_pred_norm, average='macro')
    print("f1_score " + str(f1_score))

    f1 = [f1_score]
    File = {'val':f1}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(f1_macro_score_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    f1_score = metrics.f1_score(y_true_norm, y_pred_norm, average='micro')
    print("f1_score " + str(f1_score))

    f1 = [f1_score]
    File = {'val':f1}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(f1_micro_score_filename, df.values, fmt='%s',delimiter=',')
except:
    pass
