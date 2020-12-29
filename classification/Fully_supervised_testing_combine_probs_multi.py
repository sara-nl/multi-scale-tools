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
import copy
import utils
import models

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches',nargs="+", type=int, default=[10,5])
parser.add_argument('-r', '--ROI', help='upper level ROI, multi_center or multi_region',type=str, default='upper_region')
parser.add_argument('-o', '--output_folder', help='folder where store the weights',type=str)
parser.add_argument('-i', '--input_folder', help='folder where the input data (csv files) are stored',type=str)
parser.add_argument('-n', '--N_CLASSES', help='number of classes',type=int, default=5)
parser.add_argument('-v','--verbose', help='verbose', type=bool, default=False)

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)

MAGNIFICATION = args.MAGS
print(MAGNIFICATION)
MAGNIFICATIONS_str = str(MAGNIFICATION)
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(" ", "")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(",", "_")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("[", "")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("]", "")

UPPER_REGION = args.ROI

current_UPPER = UPPER_REGION

verbose = args.verbose 

seed = int(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))

#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")

models_path = args.output_folder
#models_path = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/'
utils.create_dir(models_path)

#path model file
model_weights_filename = models_path+'multi_scale_model.pt'
model_weights_filenames_mags = [models_path+'multi_scale_model_'+str(x)+'x.pt' for x in MAGNIFICATION]

checkpoint_path = models_path+'/checkpoints/'
utils.create_dir(checkpoint_path)

#CSV LOADING
print("CSV LOADING ")
#filenames data

csv_strong_annotations = args.input_folder
#csv_strong_annotations = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/'
utils.create_dir(csv_strong_annotations)

test_dir = os.path.split(csv_strong_annotations[:-1])[0]+'/'

csv_strong_annotations = csv_strong_annotations+MAGNIFICATIONS_str+'/'

csv_filename_testing = csv_strong_annotations+'test.csv'

#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

#MODEL DEFINITION
N_CLASSES = args.N_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Parameters bag
batch_size_instance = BATCH_SIZE
num_workers = 4

import generators

params_test = {'batch_size': int(batch_size_instance),
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

testing_set = generators.Dataset_multi_scale(test_dataset, MAGNIFICATION, 'test', current_UPPER)
testing_generator_strong = data.DataLoader(testing_set, **params_test)

mode_training = 'eval'

print("testing strong labels, upper_region level")

current_UPPER='upper_region'
csv_strong_annotations = test_dir+'/upper_region_partitions/'+MAGNIFICATIONS_str+'/'

csv_filename_testing = csv_strong_annotations+'test.csv'
#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values
mode_eval = 'multi_scale'

testing_set = generators.Dataset_multi_scale(test_dataset, MAGNIFICATION, 'test', current_UPPER)
testing_generator_strong = data.DataLoader(testing_set, **params_test)

y_pred = []
y_true = []

model = torch.load(model_weights_filename)
model.eval()
model.to(device)

with torch.no_grad():
    for inputs,labels in testing_generator_strong:
        for t in inputs:
            t.to(device)

        labels_high = labels[0].to(device)
        
        
        # forward + backward + optimize
        outputs, _ = model(inputs, None, mode_eval)

        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels_high.cpu().data.numpy()
        outputs_np = np.argmax(outputs_np, axis=1)
        
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)

kappa_score_general_filename = checkpoint_path+'kappa_score_upper_region.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_upper_region.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_upper_region.csv'
f1_score_filename = checkpoint_path+'f1_score_upper_region.csv'
roc_auc_filename = checkpoint_path+'roc_auc_upper_region.csv'

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

print("testing strong labels, multicenter patch level")

current_UPPER='multi_center'
csv_strong_annotations = test_dir+'/multicenter_partitions/'+MAGNIFICATIONS_str+'/'

csv_filename_testing = csv_strong_annotations+'test.csv'
#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values
mode_eval = 'multi_scale'

testing_set = generators.Dataset_multi_scale(test_dataset, MAGNIFICATION, 'test', current_UPPER)
testing_generator_strong = data.DataLoader(testing_set, **params_test)

y_pred = []
y_true = []

model = torch.load(model_weights_filename)
model.eval()
model.to(device)

with torch.no_grad():
    for inputs,labels in testing_generator_strong:
        for t in inputs:
            t.to(device)

        labels_high = labels.to(device)
        
        
        # forward + backward + optimize
        outputs, _ = model(inputs, None, mode_eval)

        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels_high.cpu().data.numpy()
        outputs_np = np.argmax(outputs_np, axis=1)
        
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)

kappa_score_general_filename = checkpoint_path+'kappa_score_multicenter.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_multicenter.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_multicenter.csv'
f1_score_filename = checkpoint_path+'f1_score_multicenter.csv'
roc_auc_filename = checkpoint_path+'roc_auc_multicenter.csv'

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

if (UPPER_REGION=='upper_region'):

    mode_eval = 'single_scale'
    csv_strong_annotations = test_dir+'/single_scales_partitions/'

    for a in range(len(MAGNIFICATION)): 

        print("testing strong labels, patch level magnification " + str(MAGNIFICATION[a]) + "x")

        #read data
        
        csv_filename_testing = csv_strong_annotations+'magnification_'+str(MAGNIFICATION[a])+'x/test.csv'
        test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values
        testing_set = generators.Dataset_single_scale(test_dataset[:,0],test_dataset[:,1],'test')
        testing_generator_strong = data.DataLoader(testing_set, **params_test)

        y_pred = []
        y_true = []

        model = torch.load(model_weights_filenames_mags[a])
        model.eval()
        model.to(device)


        with torch.no_grad():
            for inputs,labels in testing_generator_strong:
                inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
                
                
                # forward + backward + optimize
                outputs, _ = model(inputs, a, mode_eval)

                #accumulate values
                outputs_np = outputs.cpu().data.numpy()
                labels_np = labels.cpu().data.numpy()
                outputs_np = np.argmax(outputs_np, axis=1)
                            
                y_pred = np.append(y_pred,outputs_np)
                y_true = np.append(y_true,labels_np)

        kappa_score_general_filename = checkpoint_path+'kappa_score_'+str(MAGNIFICATION[a])+'x.csv'
        acc_balanced_filename = checkpoint_path+'acc_balanced_'+str(MAGNIFICATION[a])+'x.csv'
        confusion_matrix_filename = checkpoint_path+'conf_matr_'+str(MAGNIFICATION[a])+'x.csv'
        f1_score_filename = checkpoint_path+'f1_score_'+str(MAGNIFICATION[a])+'x.csv'
        roc_auc_filename = checkpoint_path+'roc_auc_'+str(MAGNIFICATION[a])+'x.csv'

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





