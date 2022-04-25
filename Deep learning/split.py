import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'datasets/generation'

for (root, dirs, files) in os.walk(root_dir):
	for dir in dirs:
		folder = dir
		os.makedirs(root_dir +'/train/' + folder)
		os.makedirs(root_dir +'/test/' + folder)

for (root, dirs, files) in os.walk(root_dir):
	for dir in dirs:
		current = dir
		src = root_dir+'/'+current # Folder to copy images from

		allFileNames = os.listdir(src)
		np.random.shuffle(allFileNames)
		train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.7)])
		
		train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
		test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

		for name in train_FileNames:
    			shutil.copy(name, "datasets/generation/train/"+current)

		for name in test_FileNames:
    			shutil.copy(name, "datasets/generation/test/"+current)

