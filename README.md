# Multiscale tools library

Multi-scale tools library includes four components: the preprocessing tools, the regressor tool, the classification tools and the segmentation tools.
All the scripts work using csv files. At the beginning of each script, the csvs are loaded. For each component, a folder is created. The folder, besides the scripts, includes also a `utils.py`, `generators.py` and `models.py` (the latter ones only where needed).

## Reference
If you find this repository useful in your research, please cite:

Marini, N., Otálora, S., Podareanu, D., van Rijthoven, M., van der Laak, J., Ciompi, F., Müller H. & Atzori, M. (2021). Multi_Scale_Tools: a python library to exploit multi-scale whole slide images. Frontiers in Computer Science, 3, 68.


## Install
`python -m pip install git+https://github.com/sara-nl/multi-scale-tools`

## Preprocessing tools
### Description
This component includes scripts to extract the patches from several WSI's scales.
The scripts, given an WSI and the corresponding tissue mask, extract and store the patches.
The scripts developed are four, considering the two modalities developed to extract the patches and the two possible types of tissue mask used.
The tools include two possible modalities to extract the patches: the grid extraction and the multi-center extration.
In the grid extraction (scripts with *Grid* in the name), a magnification level is selected, the WSI is split in a grid (the dimension of the patches depends on the scale selected) and the patches included in tissue regions are extracted and stored.
In the multi-center extraction (scripts with *Centroids* in the name), multiple magnification levels are selected. 
The highest level is split in a grid with the same criterion presented in the previous modality.
Each patch, within the grid, becomes the centroid of a new region included in a patch at lower magnification level. 
Therefore, for each patch extracted from the highest level there is a patch, from lower magnification, that includes the patch in its center part.
The tools are developed to work with an WSI and the corresponding tissue mask, a png file generated from the WSIs at the lowest magnification available for each image, in our case at 1.25x.
The tissue mask is used to generate the coordinates for the patches, evaluating if the corresponding region includes enough tissue or background, and eventually the class of the patch.
Two different types of tissue masks are available: the masks that include only tissue region (scripts without *Strongly* in the name) and the masks annotated with the classes (scripts with *Strongly* in the name).

Where to find the scripts: `preprocessing/` folder

### How to use
An example of usage is in the file `run_scripts.sh`, in the same folder.
Four scripts are available. 
The scripts, besides the parameters to generate the grid of patches, need the following parameters:
* -i: an input csv file with the paths of the image to extract. An example of this file is in:  `csv_folder/dataset_preprocessing/`
* -t: the folder where the masks are stored. If the masks are pixel-wise annotated: The folder includes the files, with the name of the WSI+\*.png. These masks are uploaded in `Masks_ASAP`. If the masks include only the tissue region: the masks are generated using HistoQC tool. The folder includes folders with the name of the WSI. The folders include a file with the `mask_use.png` suffix. These masks are uploaded in `MASKS/`
* -o: the path of the output, where to store the images. These parameter is present also in other scripts, therefore it is important to use always the same path.
* -p, the amount the threads
* -w: the magnification level of the mask. The default is 1.25x. Do not change it
* -m: if the modality to extract the patch is grid, then the scale is a single value. If the modality to extract the patches is multicenter, is it a list.

## Scale Regressor tool
### Description
This component includes scripts to generate the input data, train, test and use a scale regressor.
The scripts that regard the regressor work at patch level: given a patch, from a certain magnification level, the regression model's output is the scale level.
Three scripts are developed: the training, the testing and the regression scripts.
The regression model is trained and tested using tuples of (image,scale level). 
Seven magnifications level are used: 5x, 8x, 10x, 15x, 20x, 30x, 40x. 
In order to have better performance between these scale levels, an augmentation scale technique is used to perturbate both the image and the label on the fly.
With these technique it is possible to modify the image, change the resolution and crop the image, but also to modify the corresponding level.
However, this scripts are made with the purpose of train and test the model, not to use it in the code.
For this reason, there is also another script called regression.py.
This script is made in order to have a method that it is possible to import and call within the code. 
The script works both standalone (it has parameters) but it is also importable.
Also if the patches are already extracted (previously presented), an intermediate step is needed to generate the input data.
The scripts to train and test the models need csv file as input data. 
The input data file includes the path of the patch and the corresponding scale level.
Two scripts to generate the input data are proposed.
The first one (`Create_csv_from_partitions`), starting from three partitions (prepared by the user) that include the names of the WSI, generate three files, the input data for training, validation and testing of the models.
The second one (`Create_csv`), starting from a list of file, generate an input data csv.

Where to find the scripts: `regressor/` folder 

### How to use
An example of usage is in the file scripts.sh, in the same folder. I*ve changed the files and the path, so that it is possible to run the scripts.
The script can be split in the groups: the input data generation, the training and the regression.

The *input data generation* scripts need:
* -i: the input folder where the partitions file or the file with the list of images are stored. An example of this folder is in: `csv_folder/scale_regression/partitions/`
* -p: the folder where the patches are stored. This should be what -o was for the preprocessing scripts.
* -o: the path of the folder where to store the csv files generated.
* -m: the magnifications wanted to be put in the csv. They will be used to train the model

The scripts *to train the model* (besides the hyperparameters):
* -i: the folder where the input data are stored
* -o: the folder where to store the model weights, some hyperparameters and the metrics file
* -c: the CNN pre-trained to use
* -vebose: if the script must be verbose or not
* -m: the magnifications used as classes

The scripts to *use the regressor*:
* -i: the input data. 
* -p: the path of the model trained. The input data can be of three different types: the path of a folder that includes the patches, the path a single patch or a csv file, that includes the path of the patches.
* -b: batch size

## Classification tools
### Description
This component includes scripts for training a CNN in a fully supervised fashion. 
Three different training variants are proposed: one training the model with patches that come from a single scale (single scale model), and two training the models with patches that come from multiple scales (the multi-scale multi-center training and the multi-scale upper region training). 
The single scale model is used as baseline. 
The input is composed of patches from magnification x and the output is the classes predicted.
In the multi-scales models, the model is fed with patches that come from several magnifications.
The models are composed of multiple CNNs, one for each scale. 
The models are trained with two combinations: combining the features of the CNN layers or the predictions made, from the different models.
The input is composed of patches from several magnifications (and the corresponding labels). Each CNN is trained with patches from a single magnification level. 
Depending on the variant adopted, the output of the model varies.

In the multi-scale multi-center, the input is composed by a tuple of patches (extracted at different levels, but with the same centroid) and the label of the patch at the highest magnification.
The output of the model is the label of the patch used as centroid, from the highest magnification level.
In this training variant, the patches used are the ones generated with the centroids methods presented in the pre-processing section. 
In the multi-scale upper-region, the input is composed by a tuple of patches (extracted at different levels, but the patches from the higher magnifications come from the same region included in the corresponding patch at lower magnification) and the labels of the patches for each magnification.
The output of the model is the label of the patch at the at highest level (obtained combining the features or the predictions) and the labels of the patches for each magnification.
In this training variant, the patches used are the ones generated with the centroids methods presented in the pre-processing section. 
In this training variant, the patches used are the ones generated with the grid methods presented in the pre-processing section. 

Also if the patches are already extracted (previously presented), an intermediate step is needed to generate the input data.
The scripts need csv file as input data. 
The input data file includes the path (or paths) of the patches and the corresponding class.
Three scripts to generate the input data are proposed (one for each training variant).
The structure is similar to the one presented for the regression: the user must create partitions for training, validation and test the models, including the WSI name.
The first one (`Generate_csv_single_scale`) generates the input data for the single scale model. It uses the partitions, the patches  and the csvs generated with the Grid scripts.
The second one (`Generate_csv_multicenter`) generates the input data for the multi-scale multicenter model. It uses the partitions, the patches  and the csvs generated with the Centroids scripts.
The second one (Generate_csv_upper_region) generates the input data for the multi-scale upper region model. It is the most time consuming script among the three. It uses the partitions, the patches  and the csvs generated with the Grid scripts. Given the magnification level wanted, from each patch from the highest resolution, a patch from lower resolution that include the first one is found. It works because during the patches extraction, a csv with the patches coordinates at the highest magnification is stored.


Where to find the scripts: `classification/` folder

### How to use
Several examples of usage are included in the folder.
run_generate_scripts.sh includes the examples to generate the input data. run_scripts_single_scale_trainining includes the examples to train the single scale model. run_scripts_multi_scale the examples to train the multi scale models, both the multi_center and the upper_region, combining both features and probabilities. I*ve changed the files and the path, so that it is possible to run the scripts.

The scripts to *generate the csvs*, besides the parameters to generate the grid of patches, need the following parameters:
* -i: the input folder where the partitions file or the file with the list of images are stored. An example of this folder is in:  `csv_folder/classification/partitions/`
* -p: the folder where the patches are stored. This should be what -o was for the preprocessing scripts.
* -o: the path of the folder where to store the csv files generated.
* -m: the magnifications wanted to be put in the csv. They will be used to train the model

The scripts *to train and test the models*:
Single scale (Fully supervised train)
* -n: the number of classes
* -i: the folder where the input data are stored
* -o: the folder where to store the model weights, some hyperparameters and the metrics file
* -m: the magnification level chosen.
* -c: the pre-trained cnn to use

Multi scale (`Fully_supervised_training_combine_probs_multi.py` and `Fully_supervised_training_combine_probs_multi.py`)

The script implements both the multi-center and the upper-region models. In order to switch between them, the parameter -r is used
* -r: the parameter to train the model using one of the two multi-scale variants. It can be: *upper_region* and *multicenter*
* -m: the magnification levels chosen.
* -c: the pre-trained cnn to use
* -i: the folder where the input data are stored
* -o: the folder where to store the model weights, some hyperparameters and the metrics file

# Acknowledgement
This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 
