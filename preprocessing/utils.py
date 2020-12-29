import os
import numpy as np
import pandas as pd

def create_dir(models_path):
	if not os.path.isdir(models_path):
		try:
			os.makedirs(models_path)
		except OSError:
			print ("Creation of the directory %s failed" % models_path)
		else:
			print ("Successfully created the directory %s " % models_path)

def check_background_weakly(glimpse,threshold,GLIMPSE_SIZE_SELECTED_LEVEL):
	b = False

	window_size = GLIMPSE_SIZE_SELECTED_LEVEL
	tot_pxl = window_size*window_size
	white_pxl = np.count_nonzero(glimpse)
	score = white_pxl/tot_pxl
	if (score>=threshold):
		b=True
	return b

def check_label_strongly(glimpse,threshold):

	label = -1

	window_size = int(glimpse.shape[0])
	tot_pxl = window_size*window_size
		#score background
	try:
		num_occurrences = np.bincount(glimpse.flatten())
		value = num_occurrences.argmax()
		n_occ_value = num_occurrences.max()
		score = n_occ_value/tot_pxl

		if (score>=threshold):

			label = convert_label(value)
	except:
		label = -1
	return label

def check_background_strongly(glimpse,threshold):

	label = None
	b = False
		#score background
	window_size = int(glimpse.shape[0])
	tot_pxl = window_size*window_size
	white_pxl = np.count_nonzero(glimpse)
	score = white_pxl/tot_pxl
	if (score>=threshold):

		num_occurrences = np.bincount(glimpse.flatten())

		value = num_occurrences.argmax()

		n_occ_value = num_occurrences.max()

		score = n_occ_value/tot_pxl

		if (score>=threshold):

			b=True
		
			label = convert_label(value)

	return b, label

#TODO: change
def convert_label(value):
	label = 0
	if (value==50):
		label = 0
	elif (value==100):
		label = 1
	elif (value==150):
		label = 2
	elif (value==200):
		label = 3
	elif (value==250):
		label = 4
	return label

def chunker_list(seq, size):
		return (seq[i::size] for i in range(size))

def available_magnifications(mpp, level_downsamples):
	mpp = float(mpp)
	if (mpp<0.26):
		magnification = 40
	else:
		magnification = 20
	
	mags = []
	for l in level_downsamples:
		mags.append(magnification/l)
	
	return mags

def find_vertical_and_analyze(wsi_np):
	pixel_stride = 10
	THRESHOLD = 0.99
	b = False
	
	half = int(wsi_np.shape[1]/2)
	h1 = half-pixel_stride
	h2 = half+pixel_stride
	   
	central_section = wsi_np[:,h1:h2]
	
	#tot_surface = 2*pixel_stride*wsi_np.shape[0]
	tot_surface = central_section.shape[0]*central_section.shape[1]
	
	unique, counts = np.unique(central_section, return_counts=True)
	
	#return central_section

	if (len(counts)==1):
		b = True
	elif (counts[0]>THRESHOLD*tot_surface):
		b = True
	return b

def left_or_right(img):
	half = int(img.shape[1]/2)
	left_img = img[:,:half]

	right_img = img[:,half:]

	unique, counts_left = np.unique(left_img, return_counts=True)
	unique, counts_right = np.unique(right_img, return_counts=True)


	b = None

	if (len(counts_left)<len(counts_right)):
		b = 'right'
	elif(len(counts_left)>len(counts_right)):
		b = 'left'
	else:
		if (counts_left[1]>counts_right[1]):
			b = 'left'
		else:
			b = 'right'

	return b

def find_horizontal_and_analyze(wsi_np):
	pixel_stride = 10
	THRESHOLD = 0.99
	b = False
	
	half = int(wsi_np.shape[0]/2)
	h1 = half-pixel_stride
	h2 = half+pixel_stride
	
	central_section = wsi_np[h1:h2,:]
	
	#tot_surface = 2*pixel_stride*wsi_np.shape[0]
	tot_surface = central_section.shape[0]*central_section.shape[1]
	
	unique, counts = np.unique(central_section, return_counts=True)

	#return central_section
	
	if (len(counts)==1):
		b = True
	elif (counts[0]>THRESHOLD*tot_surface):
		b = True
	return b

def up_or_down(img):
	
	half = int(img.shape[0]/2)
	up_img = img[:half,:]
	down_img = img[half:,:]

	unique, counts_up = np.unique(up_img, return_counts=True)
	unique, counts_down = np.unique(down_img, return_counts=True)


	b = None

	if (len(counts_up)<len(counts_down)):
		b = 'down'
	elif(len(counts_up)>len(counts_down)):
		b = 'up'
	else:
		if (counts_up[1]>counts_down[1]):
			b = 'up'
		else:
			b = 'down'

	return b

def get_input_file(filename):

	fname = os.path.split(filename)[-1]

	if ('csv' in fname):
		list_dirs = pd.read_csv(filename, sep=',', header=None).values.flatten()
	else:
		list_files = os.listdir(filename)

		list_dirs = []

		for wsi in list_files:
			if ('svs' in wsi or 'mrxs' in wsi):

				list_dirs.append(filename+wsi)

	return list_dirs

def find_border_coordinates(ROI, mask_np):

	if(ROI==False):
		start_X = 0
		start_Y = 0
		end_X = int(mask_np.shape[1])
		end_Y = int(mask_np.shape[0])

	else:
		if (not(find_vertical_and_analyze(mask_np))):
			if (not(find_horizontal_and_analyze(mask_np))):
				#ALL THE PATCHES MUST BE EXTRACTED 
				start_X = 0
				start_Y = 0
				end_X = int(mask_np.shape[1])
				end_Y = int(mask_np.shape[0])
				
			else:
				#IMAGE IS DUPLICATED ORIZONTALLY
				right_side = up_or_down(mask_np)
				
				start_X = 0
				end_X = int(mask_np.shape[1])
				
				if (right_side=='up'):
					start_Y = 0
					end_Y = int(mask_np.shape[0]/2)
				else:
					start_Y = int(mask_np.shape[0]/2)
					end_Y = int(mask_np.shape[0])  
				
		else:
			#IMAGE IS DUPLICATED VERTICALLY
			right_side = left_or_right(mask_np)
			
			start_Y = 0
			end_Y = int(mask_np.shape[0])
			
			if (right_side=='left'):
				start_X = 0
				end_X = int(mask_np.shape[1]/2)
			else:
				start_X = int(mask_np.shape[1]/2)
				end_X = int(mask_np.shape[1])

	return start_X, end_X, start_Y, end_Y

def multi_one_hot_enc(current_labels, N_CLASSES):
	labels = np.zeros(N_CLASSES)
	for i in range(len(current_labels)):
		labels[current_labels[i]]=1
	return labels

def eval_whitish_threshold(mask, thumb):
	a = np.ma.array(thumb, mask=np.logical_not(mask))
	mean_a = a.mean()

	if (mean_a<=155):
		THRESHOLD = 195.0
	elif (mean_a>155 and mean_a<=180):
		THRESHOLD = 200.0
	elif (mean_a>180):
		THRESHOLD = 205.0
	return THRESHOLD

def whitish_img(img, THRESHOLD_WHITE):
	b = True
	if (np.mean(img) > THRESHOLD_WHITE):
		b = False
	return b

def write_labels_local_file_GRID(PATH_OUTPUT,fname,arrays):
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_labels_densely.csv'
		#create file
	File = {'filename':arrays[0],'labels':arrays[1]}
	df = pd.DataFrame(File,columns=['filename','labels'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_coords_local_file_GRID(PATH_OUTPUT,fname,arrays):
		#select path
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_coords_densely.csv'
		#create file
	File = {'filename':arrays[0],'level':arrays[1],'x_top':arrays[2],'y_top':arrays[3],'magnifications':arrays[4]}
	df = pd.DataFrame(File,columns=['filename','level','x_top','y_top','magnifications'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_paths_local_file_GRID(PATH_OUTPUT,fname,listnames):
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_paths_densely.csv'
		#create file
	File = {'filenames':listnames}
	df = pd.DataFrame(File,columns=['filenames'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_coords_local_file_CENTROIDS(PATH_OUTPUT, fname, arrays, MAGNIFICATIONS):
		#select path
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	for a, m in enumerate(MAGNIFICATIONS):
		filename_path = output_dir+'/magnification_'+str(m)+'x/'+fname+'_coords_densely.csv'

			#create file
		File = {'filename':arrays[0][a],'level':arrays[1][a],'x_top':arrays[2][a],'y_top':arrays[3][a],'magnification':arrays[4][a]}
		df = pd.DataFrame(File,columns=['filename','level','x_top','y_top','magnification'])
			#save file
		np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_paths_local_file_CENTROIDS(PATH_OUTPUT, fname, listnames, MAGNIFICATIONS):
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	for a, m in enumerate(MAGNIFICATIONS):
		filename_path = output_dir+'/magnification_'+str(m)+'x/'+fname+'_paths_densely.csv'

			#create file
		File = {'filename':listnames[0][a]}
		df = pd.DataFrame(File,columns=['filename'])
			#save file
		np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_labels_local_file_CENTROIDS(PATH_OUTPUT,fname,arrays, MAGNIFICATIONS):
	output_dir = PATH_OUTPUT+'/'+fname+'/'

	for a, m in enumerate(MAGNIFICATIONS):

		filename_path = output_dir+'/magnification_'+str(m)+'x/'+fname+'_labels_densely.csv'
		#create file
		File = {'filename':arrays[0][a],'labels_higher':arrays[1][a],'labels_lower':arrays[2][a]}
		df = pd.DataFrame(File,columns=['filename','labels_higher','labels_lower'])
		#save file
		np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')


def change_magnification_name(array):
    new_array = []
    for a in array:
        
        if (a.is_integer()):
            new_array.append(int(a))
        else:
            new_array.append(a)
    
    return new_array

if __name__ == "__main__":
	pass