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

#input_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/'
input_folder = args.input_folder

csv_partition_train_filename = input_folder+'list_train.csv'
csv_partition_valid_filename = input_folder+'list_valid.csv'
csv_partition_test_filename = input_folder+'list_test.csv'

partition_train = pd.read_csv(csv_partition_train_filename, sep=',', header=None).values.flatten()
partition_valid = pd.read_csv(csv_partition_valid_filename, sep=',', header=None).values.flatten()
partition_test = pd.read_csv(csv_partition_test_filename, sep=',', header=None).values.flatten()

patches_folder = args.patches_folder
#patches_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/'

output_folder = args.output_folder
#output_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/single_scales_partitions/'
utils.create_dir(output_folder)

MAGS = args.MAGS
MAGS = utils.change_magnification_name(MAGS)
MAGS_str = str(MAGS)
MAGS_str = MAGS_str.replace(" ", "")
MAGS_str = MAGS_str.replace(",", "_")
MAGS_str = MAGS_str.replace("[", "")
MAGS_str = MAGS_str.replace("]", "")

for a, m in enumerate(MAGS): 

    out_dir = output_folder+'magnification_'+str(m)+'x/'
    utils.create_dir(out_dir)

    train_filenames = []
    train_labels = []

    valid_filenames = []
    valid_labels = []

    test_filenames = []
    test_labels = []

    for i in range(len(partition_train)):
        wsi = partition_train[i]
        local_csv = patches_folder+'magnification_'+str(m)+'x/'+wsi+'/'+wsi+'_labels_densely.csv'

        try:
            csv_labels = pd.read_csv(local_csv, sep=',', header=None).values

            for p in csv_labels:

                p_fname = p[0]
                p_label = p[1]

                train_filenames.append(p_fname)
                train_labels.append(p_label)
        except:
            pass


    for i in range(len(partition_valid)):
        wsi = partition_valid[i]
        local_csv = patches_folder+'magnification_'+str(m)+'x/'+wsi+'/'+wsi+'_labels_densely.csv'
        try:
            csv_labels = pd.read_csv(local_csv, sep=',', header=None).values

            for p in csv_labels:

                p_fname = p[0]
                p_label = p[1]

                valid_filenames.append(p_fname)
                valid_labels.append(p_label)
        except:
            pass


    for i in range(len(partition_test)):
        wsi = partition_test[i]
        local_csv = patches_folder+'magnification_'+str(m)+'x/'+wsi+'/'+wsi+'_labels_densely.csv'
        try:
            csv_labels = pd.read_csv(local_csv, sep=',', header=None).values

            for p in csv_labels:

                p_fname = p[0]
                p_label = p[1]

                test_filenames.append(p_fname)
                test_labels.append(p_label)
        except:
            pass
    
    print("MAGNIFICATION " + str(m))
    
    print(len(train_filenames), len(valid_filenames), len(test_filenames))
    
    unique, counts = np.unique(train_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    unique, counts = np.unique(valid_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    unique, counts = np.unique(test_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    
    
    new_csv_filename_train = out_dir+'train.csv'
    new_csv_filename_valid = out_dir+'valid.csv'
    new_csv_filename_test = out_dir+'test.csv'

    File = {'filename':train_filenames,'labels':train_labels}
    df = pd.DataFrame(File,columns=['filename','labels'])
    np.savetxt(new_csv_filename_train, df.values, fmt='%s',delimiter=',')

    File = {'filename':valid_filenames,'labels':valid_labels}
    df = pd.DataFrame(File,columns=['filename','labels'])
    np.savetxt(new_csv_filename_valid, df.values, fmt='%s',delimiter=',')

    File = {'filename':test_filenames,'labels':test_labels}
    df = pd.DataFrame(File,columns=['filename','labels'])
    np.savetxt(new_csv_filename_test, df.values, fmt='%s',delimiter=',')

