#!/bin/bash

#generate csv single scale
python Generate_csv_single_scale.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/single_scales_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/0/
python Generate_csv_single_scale.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/single_scales_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/1/
python Generate_csv_single_scale.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/single_scales_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/2/

#generate csv multi scale

#multicenter
#20x 10x 5x
python Generate_csv_multicenter.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/multicenter_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/0/
python Generate_csv_multicenter.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/multicenter_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/1/
python Generate_csv_multicenter.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/multicenter_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/2/

# 10x 5x
python Generate_csv_multicenter.py -m 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/multicenter_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/0/
python Generate_csv_multicenter.py -m 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/multicenter_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/1/
python Generate_csv_multicenter.py -m 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/multicenter_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_multi_magnifications_centers/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/2/

#upper region

#20x 10x 5x
python Generate_csv_upper_region.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/upper_region_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/0/
python Generate_csv_upper_region.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/upper_region_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/1/
python Generate_csv_upper_region.py -m 20 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/upper_region_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/2/

#10x 5x
python Generate_csv_upper_region.py -m 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/upper_region_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/0/
python Generate_csv_upper_region.py -m 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/upper_region_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/1/
python Generate_csv_upper_region.py -m 10 5 -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/upper_region_partitions/ -p /home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/pixel_wise_single_magnifications/ -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/partitions/2/
