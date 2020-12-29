import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import staintools
import torch.utils.data
from sklearn import metrics 
import os
import argparse 
import utils
import models


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_CLASSES', help='number classes',type=int, default=5)
parser.add_argument('-o', '--output_folder', help='folder where store the weights',type=str)
parser.add_argument('-i', '--input_folder', help='folder where the input data (csv files) are stored',type=str)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches', type=int, default=10)

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
MAGNIFICATION = args.MAGS
MAGNIFICATION_str = str(MAGNIFICATION)
print(MAGNIFICATION)

seed = int(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))

#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")

models_path = args.output_folder
#models_path = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/single_scale/'
utils.create_dir(models_path)
#path model file
model_weights_filename = models_path+'single_scale_model.pt'

checkpoint_path = models_path+'/checkpoints/'
utils.create_dir(checkpoint_path)

#CSV LOADING
print("CSV LOADING ")
#filenames data
csv_strong_annotations = args.input_folder
#csv_strong_annotations = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/single_scales_partitions/'

csv_filename_testing = csv_strong_annotations+'/magnification_'+MAGNIFICATION_str+'x/test.csv'

#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

N_CLASSES = args.N_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Parameters bag
num_workers = 4

import generators

params_test = {'batch_size': int(BATCH_SIZE),
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

testing_set = generators.Dataset_single_scale(test_dataset[:,0], test_dataset[:,1],'test')
testing_generator_strong = data.DataLoader(testing_set, **params_test)

print("testing data")

#read data

y_pred = []
y_true = []

model = torch.load(model_weights_filename)
model.eval()

with torch.no_grad():
    for inputs,labels in testing_generator_strong:
        inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
        
        
        # forward + backward + optimize
        outputs = model(inputs)

        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels.cpu().data.numpy()
        outputs_np = np.argmax(outputs_np, axis=1)
        
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)

kappa_score_general_filename = checkpoint_path+'kappa_score_general_binary_strong.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_general_binary_strong.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_general_binary_strong.csv'
f1_score_filename = checkpoint_path+'f1_score_general_binary_strong.csv'
roc_auc_filename = checkpoint_path+'roc_auc_general_binary_strong.csv'

#k-score
try:
    k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
    print("k_score " + str(k_score))

    kappas = [k_score]

    File = {'val':kappas}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(kappa_score_general_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    #confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("confusion_matrix ")
    print(str(confusion_matrix))


    conf_matr = [confusion_matrix]
    File = {'val':conf_matr}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(confusion_matrix_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    acc_balanced = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
    print("acc_balanced " + str(acc_balanced))

    acc_balancs = [acc_balanced]

    File = {'val':acc_balancs}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(acc_balanced_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    f1_score = metrics.f1_score(y_true, y_pred, average='binary')
    print("f1_score " + str(f1_score))

    f1 = [f1_score]
    File = {'val':f1}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(f1_score_filename, df.values, fmt='%s',delimiter=',')
except:
    pass

try:
    roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
    print("roc_auc_score " + str(roc_auc_score))

    roc = [roc_auc_score]
    File = {'val':roc}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(roc_auc_filename, df.values, fmt='%s',delimiter=',')
except:
    pass









