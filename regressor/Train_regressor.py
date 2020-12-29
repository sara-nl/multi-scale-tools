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
parser.add_argument('-o', '--output_folder', help='folder where store the weights and the training metadata',type=str)
parser.add_argument('-i', '--input_folder', help='folder where the input data (csv files) are stored',type=str)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches',nargs="+", type=float, default=[20,10,5])
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=False)
parser.add_argument('-d', '--dropout', help='dropout',type=float, default=0.0)
parser.add_argument('--lr', help='dropout',type=float, default=0.001)
parser.add_argument('--optimizer', help='optimizer to use', choices=['adam','sgd'], type=str, default='adam')
parser.add_argument('--loss_function', help='loss function to use', choices=['l1loss','mseloss'], type=str, default='l1loss')
parser.add_argument('-v','--verbose', help='verbose', type=bool, default=False)

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
lr = args.lr
optimizer_name = args.optimizer
loss_function = args.loss_function

verbose = args.verbose 

MAGNIFICATION = args.MAGS
MAGNIFICATION = utils.change_magnification_name(MAGNIFICATION)
MAGNIFICATIONS_str = str(MAGNIFICATION)
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(" ", "")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(",", "_")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("[", "")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("]", "")
print(MAGNIFICATION)

EMBEDDING_bool = args.features
DROPOUT = args.dropout

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

csv_filename_training = input_data_folder+'train.csv'
csv_filename_validation = input_data_folder+'valid.csv'
csv_filename_testing = input_data_folder+'test.csv'

#read data
train_dataset = pd.read_csv(csv_filename_training, sep=',', header=None).values
valid_dataset = pd.read_csv(csv_filename_validation, sep=',', header=None).values
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values
print("CSV LOADED ")

print("MODEL DEFINITION")
#MODEL DEFINITION
model = model.Regressor_model(CNN_TO_USE, EMBEDDING_bool, DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model.to(device)

print("SETTING PARAMETERS")
# Parameters bag
batch_size_instance = BATCH_SIZE
num_workers = 4

params_train = {'batch_size': batch_size_instance,
          #'shuffle': True,
          'sampler': utils.ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

params_valid = {'batch_size': batch_size_instance,
          'shuffle': True,
          'num_workers': num_workers}

params_test = {'batch_size': batch_size_instance,
          'shuffle': True,
          'num_workers': num_workers}

num_epochs = int(EPOCHS_str)

print("CREATING GENERATORS")
import generators
training_set = generators.Dataset_training(train_dataset[:,0], train_dataset[:,1], 'train', 0.5, MAGNIFICATION)
training_generator = data.DataLoader(training_set, **params_train)

validation_set = generators.Dataset_training(valid_dataset[:,0], valid_dataset[:,1], 'valid', 0.5, MAGNIFICATION)
validation_generator = data.DataLoader(validation_set, **params_valid)

testing_set = generators.Dataset_training(test_dataset[:,0], test_dataset[:,1], 'test', 0.5, MAGNIFICATION)
testing_generator_strong = data.DataLoader(testing_set, **params_test)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

if (loss_function=='mseloss'):
    criterion = torch.nn.MSELoss()
elif (loss_function=='l1loss'):
    criterion = torch.nn.L1Loss()

lr_str = str(lr)

wt_decay = 0.0
wt_decay_str = str(wt_decay)

if (optimizer_name == 'adam'):
    optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=False)
elif (optimizer_name == 'sgd'):
    optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=wt_decay, nesterov=True)



def evaluate_validation_set(epoch, generator):
    #accumulator for validation set
    y_pred_val = []
    y_true_val = []

    valid_loss = 0.0

    with torch.no_grad():
        j = 0
        for inputs,labels in generator:
            inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
            
            # zero the parameter gradients
            model.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            valid_loss = valid_loss + ((1 / (j+1)) * (loss.item() - valid_loss))   

            #accumulate values
            outputs_np = outputs.cpu().data.numpy()
            labels_np = labels.cpu().data.numpy()
                      

            del inputs, labels, outputs
            #torch.cuda.empty_cache()
            
            y_pred_val = np.append(y_pred_val,outputs_np)
            y_true_val = np.append(y_true_val,labels_np)

            if (j%100==0 and verbose==True):
                print('[%d], %d / %d loss function: %.4f' % (epoch, j, tot_batches_training, train_loss))
                explained_variance_score = metrics.explained_variance_score(y_true_val, y_pred_val)
                mean_squared_error = metrics.mean_squared_error(y_true_val, y_pred_val)
                r2_score = metrics.r2_score(y_true_val, y_pred_val)
                print("explained_variance_score " + str(explained_variance_score))
                print("mean_squared_error " + str(mean_squared_error))
                print("r2_score " + str(r2_score))

            j = j+1

        max_error = metrics.max_error(y_true_val, y_pred_val)
        explained_variance_score = metrics.explained_variance_score(y_true_val, y_pred_val)
        mean_squared_error = metrics.mean_squared_error(y_true_val, y_pred_val)
        r2_score = metrics.r2_score(y_true_val, y_pred_val)

        print('epoch [%d] valid loss: %.4f, max_error: %.4f, explained_variance_score: %.4f, mean_squared_error: %.4f, r2_score: %.4f' % (epoch, valid_loss, max_error, explained_variance_score, mean_squared_error, r2_score))

    return valid_loss


best_loss = 100000.0

losses_train = []
y_pred = []
y_true = []
    #number of epochs without improvement
EARLY_STOP_NUM = 5
early_stop_cont = 0
epoch = 0
tot_batches_training = int(len(train_dataset)/batch_size_instance)

validation_checkpoints = checkpoint_path+'validation_losses/'
utils.create_dir(validation_checkpoints)

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
#for epoch in range(num_epochs):
    #accumulator loss for the outputs
    train_loss = 0.0
    #accumulator accuracy for the outputs
    acc = 0.0
    y_pred = []
    y_true = []
    #if loss function lower
    is_best = False
    
    model.train()

    i = 0

    for inputs,labels in training_generator:
        inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
        
        # zero the parameter gradients
        model.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)

        #loss = criterion(m(outputs), labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))   

        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels.cpu().data.numpy()
        
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)
        
        if (i%100==0 and verbose==True):
            print('[%d], %d / %d loss function: %.4f' % (epoch, i, tot_batches_training, train_loss))
            explained_variance_score = metrics.explained_variance_score(y_true, y_pred)
            mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
            r2_score = metrics.r2_score(y_true, y_pred)
            print("explained_variance_score " + str(explained_variance_score))
            print("mean_squared_error " + str(mean_squared_error))
            print("r2_score " + str(r2_score))
            
        i = i+1

        del inputs, labels, outputs
        #torch.cuda.empty_cache()

    model.eval()

    print("epoch "+str(epoch)+ " train loss: " + str(train_loss))

    print("validation dataset")
    valid_loss = evaluate_validation_set(epoch, validation_generator)

    #save validation
    filename_val = validation_checkpoints+'validation_value_'+str(epoch)+'.csv'
    array_val = [valid_loss]
    File = {'val':array_val}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

    #save_hyperparameters
    filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
    array_lr = [lr_str]
    array_opt = [optimizer_name]
    array_wt_decay = [wt_decay_str]
    array_embedding = [EMBEDDING_bool]
    File = {'opt':array_opt, 'lr':array_lr,'wt_decay':array_wt_decay,'array_embedding':EMBEDDING_bool}

    df = pd.DataFrame(File,columns=['opt','lr','wt_decay','array_embedding'])
    np.savetxt(filename_hyperparameters, df.values, fmt='%s',delimiter=',')

    if (best_loss>valid_loss):
        early_stop_cont = 0
        print ("=> Saving a new best model")
        print("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
        best_loss = valid_loss
        torch.save(model, model_weights_filename)
    else:
        early_stop_cont = early_stop_cont+1
    
    epoch = epoch+1
    if (early_stop_cont == EARLY_STOP_NUM):
        print("EARLY STOPPING")

print("testing dataset")

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
               
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)

max_error_general_filename = checkpoint_path+'max_error_general_binary_strong.csv'
explained_variance_score_filename = checkpoint_path+'explained_variance_score_general_binary_strong.csv'
mean_squared_error_filename = checkpoint_path+'mean_squared_error_general_binary_strong.csv'
r2_score_filename = checkpoint_path+'r2_score_general_binary_strong.csv'

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