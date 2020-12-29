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

train_partition = pd.read_csv(csv_partition_train_filename, sep=',', header=None).values.flatten()
valid_partition = pd.read_csv(csv_partition_valid_filename, sep=',', header=None).values.flatten()
test_partition = pd.read_csv(csv_partition_test_filename, sep=',', header=None).values.flatten()

MAGS = args.MAGS
MAGS = utils.change_magnification_name(MAGS)
MAGS_str = str(MAGS)
MAGS_str = MAGS_str.replace(" ", "")
MAGS_str = MAGS_str.replace(",", "_")
MAGS_str = MAGS_str.replace("[", "")
MAGS_str = MAGS_str.replace("]", "")

patches_folder = args.patches_folder
#patches_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/'

output_folder = args.output_folder
#output_folder = '/home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/upper_region_partitions/'
output_folder = output_folder+MAGS_str+'/'
utils.create_dir(output_folder)

patches_x_folders = [patches_folder+'magnification_'+str(x)+'x/' for x in MAGS]

def get_parameters_scales(MAGNIFICATION_IN_CSV, higher_scale, lower_scale):
    
    patch_size = 224
    
    RATIOS = [MAGNIFICATION_IN_CSV / higher_scale, MAGNIFICATION_IN_CSV / lower_scale]
        
    patch_sizes = [patch_size*R for R in RATIOS]

    return patch_sizes

def lie_within(MAGNIFICATION_COORDS, higher_scale, lower_scale, x_patch_higher, y_patch_higher, csv_local_lower_level_labels, csv_local_lower_level_coords):
    
    patch_sizes = get_parameters_scales(MAGNIFICATION_COORDS, higher_scale, lower_scale)
    patch_size_high = int(patch_sizes[0])
    patch_size_low = int(patch_sizes[1])
    
    OFFSET = 0
    
    #coords patch higher level
    x_high_ini = x_patch_higher
    y_high_ini = y_patch_higher
    x_high_fin = x_high_ini+patch_size_high
    y_high_fin = y_high_ini+patch_size_high
    
    #coords center patch higher level
    x_high_c = x_high_ini+int(patch_size_high/2)
    y_high_c = y_high_ini+int(patch_size_high/2)
    
    i = 0
    b = False
    dist_max = 10000000.0
    
    filename_found = 'not found'
    label_found = -1
    
    while(b==False and i<len(csv_local_lower_level_labels)):
        
        #patch fname, patch label lower level
        p = csv_local_lower_level_labels[i]
        
        filename_p = p[0]
        label_p = p[1]
        
        #patch fname, coordinates
        c = csv_local_lower_level_coords[i]
        
        OFFSET = 0
        
        x_lower_ini = c[2]-OFFSET
        y_lower_ini = c[3]-OFFSET
        x_lower_fin = x_lower_ini+patch_size_low+OFFSET
        y_lower_fin = y_lower_ini+patch_size_low+OFFSET
        
        x_lower_c = x_lower_ini+int(patch_size_low/2)
        y_lower_c = y_lower_ini+int(patch_size_low/2)
         
        """
        dist = math.sqrt((x_lower_c - x_high_c)**2 + (y_lower_c - y_high_c)**2)
        if (dist<dist_max):
            dist_max = dist
            
            filename_found = filename_p
            label_found = label_p         
        """
        if x_lower_ini <= x_high_ini and y_lower_ini <= y_high_ini and x_lower_fin >= x_high_fin and y_lower_fin >= y_high_fin:
                       
            b = True
            filename_found = filename_p
            label_found = label_p
            
        else:
            
            i = i+1
            
    
    return filename_found, label_found

patches_x_train = [[] for m in MAGS]
labels_x_train = [[] for m in MAGS]

for wsi in train_partition:
    
    print(wsi)
    
    HIGHEST_MAGNIFICATION = MAGS[0]
    OTHER_MAGNIFICATION = MAGS[1:]
       
    csv_local_highest_level_labels_filename = patches_x_folders[0]+wsi+'/'+wsi+'_labels_densely.csv'
    csv_local_highest_level_coords_filename = patches_x_folders[0]+wsi+'/'+wsi+'_coords_densely.csv'
        
    csv_local_highest_level_labels = pd.read_csv(csv_local_highest_level_labels_filename, sep=',', header=None).values
    csv_local_highest_level_coords = pd.read_csv(csv_local_highest_level_coords_filename, sep=',', header=None).values
    
    for i in range(len(csv_local_highest_level_labels)):
        
        p = csv_local_highest_level_labels[i]
        c = csv_local_highest_level_coords[i]
        
        filename_high = p[0]
        label_high = p[1]
        
        coord_x = c[2]
        coord_y = c[3]
        magnification_coords = c[4]
        
        filenames_upper = []
        labels_upper = []
        
        j = 0
        b = True
        
        while(b==True and j<len(OTHER_MAGNIFICATION)):

            csv_local_lower_level_labels_filename = patches_x_folders[j+1]+wsi+'/'+wsi+'_labels_densely.csv'
            csv_local_lower_level_coords_filename = patches_x_folders[j+1]+wsi+'/'+wsi+'_coords_densely.csv'
    
            csv_local_lower_level_labels = pd.read_csv(csv_local_lower_level_labels_filename, sep=',', header=None).values
            csv_local_lower_level_coords = pd.read_csv(csv_local_lower_level_coords_filename, sep=',', header=None).values
                
            filename_upper, label_upper = lie_within(magnification_coords, HIGHEST_MAGNIFICATION, OTHER_MAGNIFICATION[j], coord_x, coord_y, csv_local_lower_level_labels, csv_local_lower_level_coords)
            
            filenames_upper.append(filename_upper)
            labels_upper.append(label_upper)
            
            if (label_upper==-1):
                b = False
            
            j = j+1
        
        
        if (b==True):
            patches_x_train[0].append(filename_high)
            labels_x_train[0].append(label_high)
            
            for a, m in enumerate(labels_upper):
                patches_x_train[a+1].append(filenames_upper[a])
                labels_x_train[a+1].append(labels_upper[a])
                
new_csv_filename_train = output_folder+'train.csv'

keys = []
values = []
for a, m in enumerate(MAGS):
    keys.append(str(m))
    values.append(patches_x_train[a])
    
for a, m in enumerate(MAGS):
    keys.append("label_"+str(m))
    values.append(labels_x_train[a])
    
dict_file = dict(zip(keys, values))
File = dict_file
df = pd.DataFrame(File,columns=keys)
np.savetxt(new_csv_filename_train, df.values, fmt='%s',delimiter=',')


patches_x_train = [[] for m in MAGS]
labels_x_train = [[] for m in MAGS]

for wsi in valid_partition:
    
    print(wsi)
    
    HIGHEST_MAGNIFICATION = MAGS[0]
    OTHER_MAGNIFICATION = MAGS[1:]
       
    csv_local_highest_level_labels_filename = patches_x_folders[0]+wsi+'/'+wsi+'_labels_densely.csv'
    csv_local_highest_level_coords_filename = patches_x_folders[0]+wsi+'/'+wsi+'_coords_densely.csv'
        
    csv_local_highest_level_labels = pd.read_csv(csv_local_highest_level_labels_filename, sep=',', header=None).values
    csv_local_highest_level_coords = pd.read_csv(csv_local_highest_level_coords_filename, sep=',', header=None).values
    
    for i in range(len(csv_local_highest_level_labels)):
        
        p = csv_local_highest_level_labels[i]
        c = csv_local_highest_level_coords[i]
        
        filename_high = p[0]
        label_high = p[1]
        
        coord_x = c[2]
        coord_y = c[3]
        magnification_coords = c[4]
        
        filenames_upper = []
        labels_upper = []
        
        j = 0
        b = True
        
        while(b==True and j<len(OTHER_MAGNIFICATION)):

            csv_local_lower_level_labels_filename = patches_x_folders[j+1]+wsi+'/'+wsi+'_labels_densely.csv'
            csv_local_lower_level_coords_filename = patches_x_folders[j+1]+wsi+'/'+wsi+'_coords_densely.csv'
    
            csv_local_lower_level_labels = pd.read_csv(csv_local_lower_level_labels_filename, sep=',', header=None).values
            csv_local_lower_level_coords = pd.read_csv(csv_local_lower_level_coords_filename, sep=',', header=None).values
                
            filename_upper, label_upper = lie_within(magnification_coords, HIGHEST_MAGNIFICATION, OTHER_MAGNIFICATION[j], coord_x, coord_y, csv_local_lower_level_labels, csv_local_lower_level_coords)
            
            filenames_upper.append(filename_upper)
            labels_upper.append(label_upper)
            
            if (label_upper==-1):
                b = False
            
            j = j+1
        
        
        if (b==True):
            patches_x_train[0].append(filename_high)
            labels_x_train[0].append(label_high)
            
            for a, m in enumerate(labels_upper):
                patches_x_train[a+1].append(filenames_upper[a])
                labels_x_train[a+1].append(labels_upper[a])
                
        
new_csv_filename_train = output_folder+'valid.csv'


keys = []
values = []
for a, m in enumerate(MAGS):
    keys.append(str(m))
    values.append(patches_x_train[a])
    
for a, m in enumerate(MAGS):
    keys.append("label_"+str(m))
    values.append(labels_x_train[a])
    
dict_file = dict(zip(keys, values))
File = dict_file
df = pd.DataFrame(File,columns=keys)
np.savetxt(new_csv_filename_train, df.values, fmt='%s',delimiter=',')

patches_x_train = [[] for m in MAGS]
labels_x_train = [[] for m in MAGS]

for wsi in test_partition:
    
    print(wsi)
    
    HIGHEST_MAGNIFICATION = MAGS[0]
    OTHER_MAGNIFICATION = MAGS[1:]
       
    csv_local_highest_level_labels_filename = patches_x_folders[0]+wsi+'/'+wsi+'_labels_densely.csv'
    csv_local_highest_level_coords_filename = patches_x_folders[0]+wsi+'/'+wsi+'_coords_densely.csv'
        
    csv_local_highest_level_labels = pd.read_csv(csv_local_highest_level_labels_filename, sep=',', header=None).values
    csv_local_highest_level_coords = pd.read_csv(csv_local_highest_level_coords_filename, sep=',', header=None).values
    
    for i in range(len(csv_local_highest_level_labels)):
        
        p = csv_local_highest_level_labels[i]
        c = csv_local_highest_level_coords[i]
        
        filename_high = p[0]
        label_high = p[1]
        
        coord_x = c[2]
        coord_y = c[3]
        magnification_coords = c[4]
        
        filenames_upper = []
        labels_upper = []
        
        j = 0
        b = True
        
        while(b==True and j<len(OTHER_MAGNIFICATION)):

            csv_local_lower_level_labels_filename = patches_x_folders[j+1]+wsi+'/'+wsi+'_labels_densely.csv'
            csv_local_lower_level_coords_filename = patches_x_folders[j+1]+wsi+'/'+wsi+'_coords_densely.csv'
    
            csv_local_lower_level_labels = pd.read_csv(csv_local_lower_level_labels_filename, sep=',', header=None).values
            csv_local_lower_level_coords = pd.read_csv(csv_local_lower_level_coords_filename, sep=',', header=None).values
                
            filename_upper, label_upper = lie_within(magnification_coords, HIGHEST_MAGNIFICATION, OTHER_MAGNIFICATION[j], coord_x, coord_y, csv_local_lower_level_labels, csv_local_lower_level_coords)
            
            filenames_upper.append(filename_upper)
            labels_upper.append(label_upper)
            
            if (label_upper==-1):
                b = False
            
            j = j+1
        
        
        if (b==True):
            patches_x_train[0].append(filename_high)
            labels_x_train[0].append(label_high)
            
            for a, m in enumerate(labels_upper):
                patches_x_train[a+1].append(filenames_upper[a])
                labels_x_train[a+1].append(labels_upper[a])
                
        
new_csv_filename_train = output_folder+'test.csv'

keys = []
values = []
for a, m in enumerate(MAGS):
    keys.append(str(m))
    values.append(patches_x_train[a])
    
for a, m in enumerate(MAGS):
    keys.append("label_"+str(m))
    values.append(labels_x_train[a])
    
dict_file = dict(zip(keys, values))
File = dict_file
df = pd.DataFrame(File,columns=keys)
np.savetxt(new_csv_filename_train, df.values, fmt='%s',delimiter=',')

