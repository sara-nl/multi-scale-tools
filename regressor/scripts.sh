#!/bin/bash

#create csv from partition
python Create_csv_from_partitions.py -m 40 30 20 15 10 8 5 -o FOLDER_WHERE_CSV_FILES_ARE_STORED -p FOLDER_WHERE_CSV_PATCHES_DIRECTORY_ARE_STORED -i FOLDER_WHERE_PARTITIONS_ARE_STORED
python Create_csv.py -m 40 30 20 15 10 8 5 -o FOLDER_WHERE_CSV_FILES_ARE_STORED -p FOLDER_WHERE_CSV_PATCHES_DIRECTORY_ARE_STORED -i FOLDER_WHERE_PARTITIONS_ARE_STORED

#train model
python Train_regressor.py -o FOLDER_WHERE_MODEL_FILE_IS_STORED -i FOLDER_WHERE_CSV_FILES_ARE_STORED -c resnet34 -b 128 -e 10 -m 40 30 20 15 10 8 5 -v True

#test model
python Test_regressor.py -o FOLDER_WHERE_MODEL_FILE_IS_STORED -i FOLDER_WHERE_CSV_FILES_ARE_STORED -c resnet34 -b 1024 -f test -v True
python Test_regressor.py -o FOLDER_WHERE_MODEL_FILE_IS_STORED -i FOLDER_WHERE_CSV_FILES_ARE_STORED -c resnet34 -b 1024 -f external_data -v True

#regressor to use in the code
python regressor.py -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/regressor/ -i PATH_TO_IMAGE_FILE
python regressor.py -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/regressor/ -i PATH_TO_FOLDER
python regressor.py -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/regressor/ -i PATH_TO_CSV