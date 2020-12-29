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

MAGS = args.MAGS
MAGS = utils.change_magnification_name(MAGS)
MAGS_str = str(MAGS)
MAGS_str = MAGS_str.replace(" ", "")
MAGS_str = MAGS_str.replace(",", "_")
MAGS_str = MAGS_str.replace("[", "")
MAGS_str = MAGS_str.replace("]", "")

#input_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/'
input_folder = args.input_folder

csv_partition_train_filename = input_folder+'list_train.csv'
csv_partition_valid_filename = input_folder+'list_valid.csv'
csv_partition_test_filename = input_folder+'list_test.csv'

partition_train = pd.read_csv(csv_partition_train_filename, sep=',', header=None).values.flatten()
partition_valid = pd.read_csv(csv_partition_valid_filename, sep=',', header=None).values.flatten()
partition_test = pd.read_csv(csv_partition_test_filename, sep=',', header=None).values.flatten()

patches_folder = args.patches_folder
#patches_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/'
patches_folder = patches_folder+'MAGNIFICATIONS_'+MAGS_str+'/'

output_folder = args.output_folder
#output_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/multicenter_partitions/'
output_folder = output_folder+MAGS_str+'/'
utils.create_dir(output_folder)

filenames_train = [[] for a in range(len(MAGS))]
labels_train = []
filenames_valid = [[] for a in range(len(MAGS))]
labels_valid = []
filenames_test = [[] for a in range(len(MAGS))]
labels_test = []


for i in range(len(partition_train)):
    wsi = partition_train[i]
    
    for a, m in enumerate(MAGS):
        filenames_x_prefix = patches_folder+wsi+'/magnification_'+str(MAGS[a])+'x/'
        local_csv_filename = filenames_x_prefix+wsi+'_labels_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values
        
    
        for l in csv_local:
            
            if (a==0):
            
                filenames_train[a].append(l[0])
                labels_train.append(l[1])
            
            else:
                
                filenames_train[a].append(l[0])
                
                
for i in range(len(partition_valid)):
    wsi = partition_valid[i]
    
    for a, m in enumerate(MAGS):
        filenames_x_prefix = patches_folder+wsi+'/magnification_'+str(MAGS[a])+'x/'
        local_csv_filename = filenames_x_prefix+wsi+'_labels_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values
        
    
        for l in csv_local:
                        
            if (a==0):
            
                filenames_valid[a].append(l[0])
                labels_valid.append(l[1])
            
            else:
                
                filenames_valid[a].append(l[0])           
                
for i in range(len(partition_test)):
    wsi = partition_test[i]
    
    for a, m in enumerate(MAGS):
        filenames_x_prefix = patches_folder+wsi+'/magnification_'+str(MAGS[a])+'x/'
        local_csv_filename = filenames_x_prefix+wsi+'_labels_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values
        
    
        for l in csv_local:
            
            if (a==0):
            
                filenames_test[a].append(l[0])
                labels_test.append(l[1])
            
            else:
                
                filenames_test[a].append(l[0])

print(len(filenames_train), len(filenames_valid), len(filenames_test))
    
unique, counts = np.unique(labels_train, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(labels_valid, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(labels_test, return_counts=True)
print(dict(zip(unique, counts)))

new_csv_filename_train = output_folder+'train.csv'
new_csv_filename_valid = output_folder+'valid.csv'
new_csv_filename_test = output_folder+'test.csv'


keys = []
values = []
for a, m in enumerate(MAGS):
    keys.append(str(m))
    values.append(filenames_train[a])

keys.append('labels')
values.append(labels_train)

dict_file = dict(zip(keys, values))
File = dict_file
df = pd.DataFrame(File,columns=keys)
np.savetxt(new_csv_filename_train, df.values, fmt='%s',delimiter=',')


keys = []
values = []
for a, m in enumerate(MAGS):
    keys.append(str(m))
    values.append(filenames_valid[a])

keys.append('labels')
values.append(labels_valid)

dict_file = dict(zip(keys, values))
File = dict_file
df = pd.DataFrame(File,columns=keys)
np.savetxt(new_csv_filename_valid, df.values, fmt='%s',delimiter=',')


keys = []
values = []
for a, m in enumerate(MAGS):
    keys.append(str(m))
    values.append(filenames_test[a])

keys.append('labels')
values.append(labels_test)

dict_file = dict(zip(keys, values))
File = dict_file
df = pd.DataFrame(File,columns=keys)
np.savetxt(new_csv_filename_test, df.values, fmt='%s',delimiter=',')

