#!/bin/bash

#generate patches

python Patch_Extractor_Dense_Grid_Strong_Labels.py -m 10 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /projects/0/examode/experiments_HESSO/Multi_Scale_tools/csv_folder/dataset_preprocessing/annotated_images_paths.csv -t /projects/0/examode/experiments_HESSO/Multi_Scale_tools/Masks_ASAP/ -o /projects/0/examode/experiments_HESSO/Multi_Scale_tools/patches/
python Patch_Extractor_Dense_Grid.py -m 10 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /projects/0/examode/experiments_HESSO/Multi_Scale_tools/csv_folder/dataset_preprocessing/csv_to_extract.csv -t /projects/0/examode/experiments_HESSO/Multi_Scale_tools/MASKS/ -o /projects/0/examode/experiments_HESSO/patches/

python Patch_Extractor_Dense_Centroids_Strong_Labels.py -m 10 5 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /projects/0/examode/experiments_HESSO/Multi_Scale_tools/csv_folder/dataset_preprocessing/annotated_images_paths.csv -t /projects/0/examode/experiments_HESSO/Multi_Scale_tools/Masks_ASAP/ -o /projects/0/examode/experiments_HESSO/Multi_Scale_tools/patches/
python Patch_Extractor_Dense_Centroids.py -m 10 5 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /projects/0/examode/experiments_HESSO/Multi_Scale_tools/csv_folder/dataset_preprocessing/csv_to_extract.csv -t /projects/0/examode/experiments_HESSO/Multi_Scale_tools/MASKS/ -o /projects/0/examode/experiments_HESSO/patches/

