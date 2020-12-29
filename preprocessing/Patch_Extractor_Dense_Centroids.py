import sys, os
import openslide
from PIL import Image
import numpy as np
import pandas as pd 
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import threading
import time
import collections
import cv2
import albumentations as A
import time
from skimage import exposure
import json
import utils 
import argparse 

np.random.seed(0)

parser = argparse.ArgumentParser(description='Configurations of the parameters for the extraction')
parser.add_argument('-m', '--MAGS', help='wanted magnification to extract the patches',nargs="+", type=int, default=[10,5])
parser.add_argument('-w', '--MASK_LEVEL', help='wanted magnification to extract the patches',type=float, default=1.25)
parser.add_argument('-i', '--INPUT_DATA', help='input data: it can be a .csv file with the paths or a directory',type=str, default='/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/csv_folder/partitions.csv')
parser.add_argument('-t', '--INPUT_MASKS', help='directory where the masks are stored',type=str, default='/home/niccolo/ExamodePipeline/Multiple_Instance_Learning/Colon/MASKS/')
parser.add_argument('-o', '--PATH_OUTPUT', help='directory where the patches will be stored',type=str, default='/home/niccolo/ExamodePipeline/Multi_Scale_Tools/patches/')
parser.add_argument('-p', '--THREADS', help='amount of threads to use',type=int, default=10)
parser.add_argument('-r', '--ROI', help='if the WSI is composed of similar slices, only one of them is used (True) or all of them (False)',type=bool, default=False)
parser.add_argument('-s', '--SIZE', help='patch size',type=int, default=224)
parser.add_argument('-x', '--THRESHOLD', help='threshold of tissue pixels to select a patches',type=float, default=0.7)
parser.add_argument('-y', '--STRIDE', help='pixel_stride between patches',type=int, default=0)

args = parser.parse_args()

MAGNIFICATIONS = args.MAGS
MAGNIFICATIONS_str = str(MAGNIFICATIONS)
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(" ", "")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace(",", "_")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("[", "_")
MAGNIFICATIONS_str = MAGNIFICATIONS_str.replace("]", "_")

MASK_LEVEL = args.MASK_LEVEL

LIST_FILE = args.INPUT_DATA

PATH_INPUT_MASKS = args.INPUT_MASKS

PATH_OUTPUT = args.PATH_OUTPUT
utils.create_dir(PATH_OUTPUT)
PATH_OUTPUT = PATH_OUTPUT+'multi_magnifications_centers/'
utils.create_dir(PATH_OUTPUT)
PATH_OUTPUT = PATH_OUTPUT+'MAGNIFICATIONS_'+MAGNIFICATIONS_str+'/'
utils.create_dir(PATH_OUTPUT)

new_patch_size = args.SIZE

THREAD_NUMBER = args.THREADS
lockList = threading.Lock()
lockGeneralFile = threading.Lock()

ROI = args.ROI
THRESHOLD = args.THRESHOLD

def generate_parameters(WANTED_LEVEL, mags, WINDOW_WANTED_LEVEL):
	#SELECTED_LEVEL = int(float(wsi.properties['aperio.AppMag']))

	#SELECTED_LEVEL = select_nearest_magnification(WANTED_LEVEL,mags)
	SELECTED_LEVEL = mags[0]
	MAGNIFICATION_RATIO = SELECTED_LEVEL/MASK_LEVEL

	GLIMPSE_SIZE_SELECTED_LEVEL = WINDOW_WANTED_LEVEL*SELECTED_LEVEL/WANTED_LEVEL
	GLIMPSE_SIZE_SELECTED_LEVEL = int(GLIMPSE_SIZE_SELECTED_LEVEL)

	GLIMPSE_SIZE_MASK = GLIMPSE_SIZE_SELECTED_LEVEL/MAGNIFICATION_RATIO
	GLIMPSE_SIZE_MASK = int(GLIMPSE_SIZE_MASK)

	STRIDE_SIZE_MASK = 0
	TILE_SIZE_MASK = GLIMPSE_SIZE_MASK+STRIDE_SIZE_MASK
	
	return GLIMPSE_SIZE_MASK, GLIMPSE_SIZE_SELECTED_LEVEL, MAGNIFICATION_RATIO
	

#estrae glimpse e salva metadati relativi al glimpse
def analyze_file(filename):
	global filename_list_general, labels_multiclass_general, labels_binary_general, csv_binary, csv_multiclass, MAGNIFICATION

	patches = []

	file = openslide.open_slide(filename)
	mpp = file.properties['openslide.mpp-x']

	level_downsamples = file.level_downsamples
	mags = utils.available_magnifications(mpp, level_downsamples)

	wanted_levels = MAGNIFICATIONS
	level = 0
		#load file
	#file = openslide.OpenSlide(filename)

		#load mask
	fname = os.path.split(filename)[-1]
		#check if exists
	fname_mask = PATH_INPUT_MASKS+fname+'/'+fname+'_mask_use.png' 

	array_dict = []

		#level 0 for the conversion
	WANTED_LEVEL = wanted_levels[0]
	HIGHEST_LEVEL = mags[0]
	#AVAILABLE_LEVEL = select_nearest_magnification(WANTED_LEVEL, mags, level_downsamples)
	
	RATIO_WANTED_MASK = WANTED_LEVEL/MASK_LEVEL
	RATIO_HIGHEST_MASK = HIGHEST_LEVEL/MASK_LEVEL

	WINDOW_WANTED_LEVEL = new_patch_size

	GLIMPSE_SIZE_SELECTED_LEVEL = WINDOW_WANTED_LEVEL

	GLIMPSE_SIZE_MASK = np.around(GLIMPSE_SIZE_SELECTED_LEVEL/RATIO_WANTED_MASK)
	GLIMPSE_SIZE_MASK = int(GLIMPSE_SIZE_MASK)

	GLIMPSE_HIGHEST_LEVEL = np.around(GLIMPSE_SIZE_MASK*RATIO_HIGHEST_MASK)
	GLIMPSE_HIGHEST_LEVEL = int(GLIMPSE_HIGHEST_LEVEL)
	
	STRIDE_SIZE_MASK = 0
	TILE_SIZE_MASK = GLIMPSE_SIZE_MASK+STRIDE_SIZE_MASK

	output_dir = PATH_OUTPUT+fname

	#if (os.path.isfile(fname_mask) and os.path.isdir(output_dir)):
	if (os.path.isfile(fname_mask)):
			#creates directory
		output_dir = PATH_OUTPUT+fname
		utils.create_dir(output_dir)

		output_dir_m = []

		patches = []

		for m in MAGNIFICATIONS:
			subdir_m = output_dir+'/magnification_'+str(m)+'x/'
			output_dir_m.append(subdir_m)
			utils.create_dir(subdir_m)

			#create CSV file structure (local)
		filename_list = [[] for m in MAGNIFICATIONS]
		level_list = [[] for m in MAGNIFICATIONS]
		x_list = [[] for m in MAGNIFICATIONS]
		y_list = [[] for m in MAGNIFICATIONS]
		magnification_patches = [[] for m in MAGNIFICATIONS]

		img = Image.open(fname_mask)

		thumb = file.get_thumbnail(img.size)
		thumb = thumb.resize(img.size)
		mask_np = np.asarray(thumb)
		img = np.asarray(img)

		mask_3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
		
		WHITISH_THRESHOLD = utils.eval_whitish_threshold(mask_3d, mask_np)

		mask_np = np.asarray(img)

		start_X, end_X, start_Y, end_Y = utils.find_border_coordinates(ROI, mask_np) 

		n_image = 0

		y_ini = start_Y + STRIDE_SIZE_MASK
		y_end = y_ini + GLIMPSE_SIZE_MASK

		while(y_end<end_Y):
	
			x_ini = start_X + STRIDE_SIZE_MASK
			x_end = x_ini + GLIMPSE_SIZE_MASK
			
			while(x_end<end_X):
				glimpse = mask_np[y_ini:y_ini+GLIMPSE_SIZE_MASK,x_ini:x_ini+GLIMPSE_SIZE_MASK]

				check_flag = utils.check_background_weakly(glimpse,THRESHOLD,TILE_SIZE_MASK)
				
				if(check_flag):

					fname_patch = output_dir_m[0]+'/'+fname+'_'+str(n_image)+'.png'
						#change to magnification 40x
					center_x = x_ini+round(GLIMPSE_SIZE_MASK/2)
					center_y = y_ini+round(GLIMPSE_SIZE_MASK/2)

					x_coords_0 = int(x_ini*RATIO_HIGHEST_MASK)
					y_coords_0 = int(y_ini*RATIO_HIGHEST_MASK)
						
					patch_high = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_HIGHEST_LEVEL,GLIMPSE_HIGHEST_LEVEL))
					patch_high = patch_high.convert("RGB")
					
					save_im = patch_high.resize((new_patch_size,new_patch_size))
					save_im = np.asarray(save_im)	

					bool_white = utils.whitish_img(save_im,WHITISH_THRESHOLD)
					bool_exposure = exposure.is_low_contrast(save_im)


					if (bool_white):
						if bool_exposure==False:
						#if (exposure.is_low_contrast(save_im)==False):

							io.imsave(fname_patch, save_im)
							
							#add to arrays (local)
							filename_list[0].append(fname_patch)
							level_list[0].append(level)
							x_list[0].append(x_coords_0)
							y_list[0].append(y_coords_0)
							magnification_patches[0].append(HIGHEST_LEVEL)

							for a, m in enumerate(wanted_levels[1:], start = 1):
								GLIMPSE_SIZE_MASK_LEVEL, GLIMPSE_SIZE_LEVEL, MAGNIFICATION_RATIO_LEVEL = generate_parameters(m, mags, WINDOW_WANTED_LEVEL)
								y_ini_level = center_y - round(GLIMPSE_SIZE_MASK_LEVEL/2)
								y_end_level = y_ini_level + GLIMPSE_SIZE_MASK_LEVEL
								
								x_ini_level = center_x - round(GLIMPSE_SIZE_MASK_LEVEL/2)
								x_end_level = x_ini_level + GLIMPSE_SIZE_MASK_LEVEL
								
								x_coords_0 = int(x_ini_level*MAGNIFICATION_RATIO_LEVEL)
								y_coords_0 = int(y_ini_level*MAGNIFICATION_RATIO_LEVEL)
								
								patch_high = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_SIZE_LEVEL,GLIMPSE_SIZE_LEVEL))
								patch_high = patch_high.convert("RGB")
								save_im = patch_high.resize((new_patch_size,new_patch_size))
								save_im = np.asarray(save_im)

								fname_patch = output_dir_m[a]+fname+'_'+str(n_image)+'.png'

								io.imsave(fname_patch, save_im)
								filename_list[a].append(fname_patch)
								level_list[a].append(level)
								x_list[a].append(x_coords_0)
								y_list[a].append(y_coords_0)
								magnification_patches[a].append(HIGHEST_LEVEL)

							n_image = n_image+1
							#save the image
							#create_output_imgs(file_10x,fname)
						else:
							print("low_contrast " + str(output_dir))

				x_ini = x_end + STRIDE_SIZE_MASK
				x_end = x_ini + GLIMPSE_SIZE_MASK

			y_ini = y_end + STRIDE_SIZE_MASK
			y_end = y_ini + GLIMPSE_SIZE_MASK
		
			#add to general arrays
		if (n_image!=0):
			lockGeneralFile.acquire()
			filename_list_general.append(output_dir)

			print("len filename " + str(len(filename_list_general)) + "; WSI done: " + filename)
			print("extracted " + str(n_image) + " patches")
			lockGeneralFile.release()
			utils.write_coords_local_file_CENTROIDS(PATH_OUTPUT,fname,[filename_list,level_list,x_list,y_list, magnification_patches],MAGNIFICATIONS)
			utils.write_paths_local_file_CENTROIDS(PATH_OUTPUT,fname,filename_list,MAGNIFICATIONS)
		else:
			print("ZERO OCCURRENCIES " + str(output_dir))

	else:
		print("no mask")

def explore_list(list_dirs):
	global list_dicts, n, csv_binary, csv_multiclass
	#print(threadname + str(" started"))

	for i in range(len(list_dirs)):
		analyze_file(list_dirs[i])
	#print(threadname + str(" finished"))

def main():
	#create output dir if not exists
	start_time = time.time()
	global list_dicts, n, filename_list_general, labels_multiclass_general, labels_binary_general, csv_binary, csv_multiclass


		#create CSV file structure (global)
	filename_list_general = []

	n = 0

	list_dirs = utils.get_input_file(LIST_FILE)
	
		#split in chunks for the threads
	list_dirs = list(utils.chunker_list(list_dirs,THREAD_NUMBER))
	print(len(list_dirs))

	threads = []
	for i in range(THREAD_NUMBER):
		t = threading.Thread(target=explore_list,args=([list_dirs[i]]))
		threads.append(t)

	for t in threads:
		t.start()
		#time.sleep(60)

	for t in threads:
		t.join()
	
		#prepare data
	
	elapsed_time = time.time() - start_time
	print("elapsed time " + str(elapsed_time))


if __name__ == "__main__":
	main()
