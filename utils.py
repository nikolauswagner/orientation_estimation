import numpy as np
import os
import json
from keras_preprocessing.image import load_img, img_to_array
from scipy.spatial.transform import Rotation
import tensorflow as tf

from PIL import Image
import cv2

# Precent OOM errors on smaller GPUs
def setupGPU():
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	tf.compat.v1.keras.backend.set_session(sess)

# TODO: remove sets_to_read parameter
def readRenderData(img_dir, sets_to_read=-1):
	X = np.empty((0, 32, 32, 3), dtype=np.uint8)
	y = np.empty((0, 3), dtype=np.float)

	for i_set, img_set in enumerate(sorted(os.listdir(img_dir))):
		if i_set == sets_to_read:
			break
		print("Processing set #" + str(i_set))
		X = np.append(X, np.empty((36*36, 32, 32, 3), dtype=np.uint8), axis=0)
		y = np.append(y, np.empty((36*36, 3), dtype=np.float), axis=0)
		for i_roll, roll in enumerate(range(0, 360, 10)):
			for i_pitch, pitch in enumerate(range(0, 360, 10)):
				# Read RGB image
				filename = img_dir + img_set + "/roll" + str(roll).zfill(3) + "_pitch" + str(pitch).zfill(3) + ".png"
				#img = load_img(filename)
				#img = img_to_array(img, dtype=np.uint8)
				img = cv2.imread(filename)
				X[i_set*36*36 + i_roll*36 + i_pitch] = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

				# Obtain directional vector from roll&pitch
				M_x = Rotation.from_euler('x', 0, degrees=True)
				M_y = Rotation.from_euler('y', roll, degrees=True)
				M_z = Rotation.from_euler('z', -pitch, degrees=True)
				M = M_x * M_y * M_z
				y[i_set*36*36 + i_roll*36 + i_pitch] = M.apply([-1,0,0])

	return X, y


def readCocoLikeData(data_dir, use_color=True, use_depth=False, interpolation=cv2.INTER_LANCZOS4):
	print('Reading dataset from ' + data_dir + 'dataset.json')
	json_file = open(data_dir + 'dataset.json', 'r')
	dataset = json.load(json_file)

	img_channels = 0
	if use_color:
		img_channels += 3
	if use_depth:
		img_channels += 1

	X = np.empty((len(dataset['annotations']), 32, 32, img_channels), dtype=np.uint8)
	y = np.empty((len(dataset['annotations']), 3), dtype=np.float)

	annotation_id = 0
	for img_data in dataset['images']:
		rgb_filename   = img_data['file_name']
		depth_filename = "depth" + rgb_filename[3:]
		img_rgb   = cv2.imread(data_dir + rgb_filename)
		img_depth = cv2.imread(data_dir + depth_filename, cv2.IMREAD_ANYDEPTH)

		for annotation in dataset['annotations']:
			if annotation['image_id'] == img_data['id']:
				x_bbox, y_bbox, w_bbox, h_bbox = annotation['bbox']
				bbox = []
				if use_color and use_depth:
					bbox_rgb   = img_rgb[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
					bbox_depth = img_depth[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
					bbox_depth = (bbox_depth/256).astype('uint8')
					bbox = np.append(bbox_rgb, bbox_depth[..., np.newaxis], axis=-1)

				elif use_color:
					bbox = img_rgb[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]

				elif use_depth:
					bbox = img_depth[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
					bbox = (bbox/256).astype('uint8')

				if interpolation:
					bbox = cv2.resize(bbox, dsize=(32, 32), interpolation=interpolation)
				else:
					border_v = 0
					border_h = 0
					if  bbox.shape[0] / bbox.shape[1] < 1:
						border_v = int(((bbox.shape[1]) - bbox.shape[0]) / 2)
					else:
						border_h = int(((bbox.shape[0]) - bbox.shape[1]) / 2)
					bbox = cv2.copyMakeBorder(bbox, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
					bbox = cv2.resize(bbox, (32, 32))

				if img_channels == 1:
					X[annotation_id] = bbox[..., np.newaxis]
				else:
					X[annotation_id] = bbox
				y[annotation_id] = np.array(annotation['orientation'])
				annotation_id = annotation_id + 1

	X = X[:annotation_id,...]
	y = y[:annotation_id,...]
	return X, y