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
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=64)
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches', type=int, default=10)
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=False)
parser.add_argument('-d', '--dropout', help='dropout',type=float, default=0.0)
parser.add_argument('--lr', help='dropout',type=float, default=0.001)
parser.add_argument('--optimizer', help='optimizer to use', choices=['adam','sgd'], type=str, default='adam')
parser.add_argument('-v','--verbose', help='verbose', type=bool, default=False)

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
MAGNIFICATION = args.MAGS
MAGNIFICATION_str = str(MAGNIFICATION)
print(MAGNIFICATION)
EMBEDDING_bool = args.features
DROPOUT = args.dropout
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

csv_filename_training = csv_strong_annotations+'/magnification_'+MAGNIFICATION_str+'x/train.csv'
csv_filename_validation = csv_strong_annotations+'/magnification_'+MAGNIFICATION_str+'x/valid.csv'
csv_filename_testing = csv_strong_annotations+'/magnification_'+MAGNIFICATION_str+'x/test.csv'

#read data
train_dataset = pd.read_csv(csv_filename_training, sep=',', header=None).values
valid_dataset = pd.read_csv(csv_filename_validation, sep=',', header=None).values
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

N_CLASSES = args.N_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

model = models.Single_Scale_Model(CNN_TO_USE, EMBEDDING_bool, DROPOUT, N_CLASSES)
#model = freeze_unfreeze(True, model)
model.to(device)

# Parameters bag
num_workers = 4

import generators

params_train = {'batch_size': BATCH_SIZE,
          #'shuffle': True,
          'sampler': utils.ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

params_valid = {'batch_size': int(BATCH_SIZE),
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

params_test = {'batch_size': int(BATCH_SIZE),
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

num_epochs = int(EPOCHS_str)

training_set = generators.Dataset_single_scale(train_dataset[:,0], train_dataset[:,1],'train')
training_generator = data.DataLoader(training_set, **params_train)

validation_set = generators.Dataset_single_scale(valid_dataset[:,0], valid_dataset[:,1],'valid')
validation_generator = data.DataLoader(validation_set, **params_valid)

testing_set = generators.Dataset_single_scale(test_dataset[:,0], test_dataset[:,1],'test')
testing_generator_strong = data.DataLoader(testing_set, **params_test)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

criterion = torch.nn.CrossEntropyLoss()

import torch.optim as optim
optimizer_str = args.optimizer
lr = args.lr

lr_str = str(lr)

wt_decay = 0.0
wt_decay_str = str(wt_decay)


if (optimizer_str == 'adam'):
    optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=False)
elif (optimizer_str == 'sgd'):
    optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=wt_decay, nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def evaluate_validation_set(epoch, generator):
    #accumulator for validation set
    y_pred_val = []
    y_true_val = []

    valid_loss = 0.0


    with torch.no_grad():
        j = 0
        for inputs,labels in generator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            model.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            valid_loss = valid_loss + ((1 / (j+1)) * (loss.item() - valid_loss))   

            #accumulate values
            outputs_np = outputs.cpu().data.numpy()
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.argmax(outputs_np, axis=1)
                      

            del inputs, labels, outputs
            #torch.cuda.empty_cache()
            
            y_pred_val = np.append(y_pred_val,outputs_np)
            y_true_val = np.append(y_true_val,labels_np)

            j = j+1

        acc_valid = metrics.accuracy_score(y_true=y_true_val, y_pred=y_pred_val)
        acc_balanced_valid = metrics.balanced_accuracy_score(y_true=y_true_val, y_pred=y_pred_val, sample_weight=None, adjusted=False)
        kappa_valid =  metrics.cohen_kappa_score(y_true_val,y_pred_val, weights='quadratic')

        print('epoch [%d] valid loss: %.4f, acc_valid: %.4f, acc_balanced: %.4f, kappa: %.4f' % (epoch, valid_loss, acc_valid, acc_balanced_valid, kappa_valid))

        confusion_matrix = metrics.confusion_matrix(y_true=y_true_val, y_pred=y_pred_val)
        print(str(confusion_matrix))

    return valid_loss


best_loss = 100000.0

losses_train = []

    #number of epochs without improvement
EARLY_STOP_NUM = EPOCHS-2
early_stop_cont = 0
epoch = 0
tot_batches_training = int(len(train_dataset)/BATCH_SIZE)

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
        inputs, labels = inputs.to(device), labels.to(device)
        
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
        outputs_np = np.argmax(outputs_np, axis=1)
        
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)
        
        if (i%100==0 and verbose==True):
            print('[%d], %d / %d loss function: %.4f' % (epoch, i, tot_batches_training, train_loss))
            print("accuracy " + str(metrics.accuracy_score(y_true, y_pred)))
            print("kappa " + str(metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')))

        i = i+1

        del inputs, labels, outputs
        #torch.cuda.empty_cache()

    model.eval()
    scheduler.step()

    print("epoch "+str(epoch)+ " train loss: " + str(train_loss))

    print("evaluating validation")
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
    array_opt = [optimizer_str]
    array_wt_decay = [wt_decay_str]
    array_embedding = [EMBEDDING_bool]
    array_mag = [MAGNIFICATION_str]
    File = {'opt':array_opt, 'lr':array_lr,'wt_decay':array_wt_decay,'array_embedding':array_embedding, 'array_mag': array_mag}

    df = pd.DataFrame(File,columns=['opt','lr','wt_decay','array_embedding', 'array_mag'])
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

print("testing strong labels, patch level")

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









