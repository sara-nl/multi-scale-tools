import numpy as np
import pandas as pd
import random
import argparse
import utils

parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-i', '--input_folder', help='partition_folders', type=str)
parser.add_argument('-p', '--patches_folder', help='partition_folders', type=str)
parser.add_argument('-o', '--output_folder', help='partition_folders', type=str)
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches',nargs="+", type=float, default=[10,5])
args = parser.parse_args()

#input_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/scale_regression/'
input_folder = args.input_folder

csv_partition_train_filename = input_folder+'/partitions/list_train.csv'
csv_partition_valid_filename = input_folder+'/partitions/list_valid.csv'
csv_partition_test_filename = input_folder+'/partitions/list_test.csv'

partition_train = pd.read_csv(csv_partition_train_filename, sep=',', header=None).values.flatten()
partition_valid = pd.read_csv(csv_partition_valid_filename, sep=',', header=None).values.flatten()
partition_test = pd.read_csv(csv_partition_test_filename, sep=',', header=None).values.flatten()

patches_folder = args.patches_folder
#patches_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/'

output_folder = args.output_folder
#output_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/regression/training_data/'
utils.create_dir(output_folder)

MAGS = args.MAGS
MAGS = utils.change_magnification_name(MAGS)
MAGS_str = str(MAGS)
MAGS_str = MAGS_str.replace(" ", "")
MAGS_str = MAGS_str.replace(",", "_")
MAGS_str = MAGS_str.replace("[", "")
MAGS_str = MAGS_str.replace("]", "")


filenames_train = []
labels_train = []
filenames_valid = []
labels_valid = []
filenames_test = []
labels_test = []


for i in range(len(partition_train)):
    wsi = partition_train[i]
    
    for a, m in enumerate(MAGS):
        
        filenames_x_prefix = patches_folder+'/magnification_'+str(MAGS[a])+'x/'+wsi+'/'
        local_csv_filename = filenames_x_prefix+wsi+'_labels_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values[:,0]
               
        for l in csv_local:
                        
            filenames_train.append(l)
            labels_train.append(float(MAGS[a]))
            
                
                
for i in range(len(partition_valid)):
    wsi = partition_valid[i]
    
    for a, m in enumerate(MAGS):
        filenames_x_prefix = patches_folder+'/magnification_'+str(MAGS[a])+'x/'+wsi+'/'
        local_csv_filename = filenames_x_prefix+wsi+'_labels_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values[:,0]
        
    
        for l in csv_local:
            
            filenames_valid.append(l)
            labels_valid.append(float(MAGS[a]))
                            
for i in range(len(partition_test)):
    wsi = partition_test[i]
    
    for a, m in enumerate(MAGS):
        filenames_x_prefix = patches_folder+'/magnification_'+str(MAGS[a])+'x/'+wsi+'/'
        local_csv_filename = filenames_x_prefix+wsi+'_labels_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values[:,0]
        
    
        for l in csv_local:
            
            filenames_test.append(l)
            labels_test.append(float(MAGS[a]))
            

unique, counts = np.unique(labels_train, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(labels_valid, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(labels_test, return_counts=True)
print(dict(zip(unique, counts)))

new_csv_filename_train = output_folder+'train.csv'
new_csv_filename_valid = output_folder+'valid.csv'
new_csv_filename_test = output_folder+'test.csv'

File = {'filename':filenames_train,'labels':labels_train}
df = pd.DataFrame(File,columns=['filename','labels'])
np.savetxt(new_csv_filename_train, df.values, fmt='%s',delimiter=',')

File = {'filename':filenames_valid,'labels':labels_valid}
df = pd.DataFrame(File,columns=['filename','labels'])
np.savetxt(new_csv_filename_valid, df.values, fmt='%s',delimiter=',')

File = {'filename':filenames_test,'labels':labels_test}
df = pd.DataFrame(File,columns=['filename','labels'])
np.savetxt(new_csv_filename_test, df.values, fmt='%s',delimiter=',')



