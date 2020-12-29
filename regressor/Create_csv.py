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

csv_partition_filename = input_folder+'/partitions/external_dataset_source.csv'

data = pd.read_csv(csv_partition_filename, sep=',', header=None).values.flatten()

patches_folder = args.patches_folder
#patches_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/TCGA_PRAD/'

output_folder = args.output_folder
#output_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/regression/trainin_data/'
utils.create_dir(output_folder)

MAGS = args.MAGS
MAGS = utils.change_magnification_name(MAGS)
MAGS_str = str(MAGS)
MAGS_str = MAGS_str.replace(" ", "")
MAGS_str = MAGS_str.replace(",", "_")
MAGS_str = MAGS_str.replace("[", "")
MAGS_str = MAGS_str.replace("]", "")


filenames = []
labels = []


for i in range(len(data)):
    wsi = data[i]
    
    for a, m in enumerate(MAGS):
        
        filenames_x_prefix = patches_folder+'/magnification_'+str(MAGS[a])+'x/'+wsi+'/'
        local_csv_filename = filenames_x_prefix+wsi+'_coords_densely.csv'
        csv_local = pd.read_csv(local_csv_filename, sep=',', header=None).values[:,0]
               
        for l in csv_local:
                        
            filenames.append(l)
            labels.append(float(MAGS[a]))        

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

new_csv_filename = output_folder+'external_data.csv'

File = {'filename':filenames,'labels':labels}
df = pd.DataFrame(File,columns=['filename','labels'])
np.savetxt(new_csv_filename, df.values, fmt='%s',delimiter=',')




