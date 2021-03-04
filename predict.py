import numpy as np 

from keras import backend as K
from keras import models
from keras import preprocessing
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from regression import PoseRegressor, RotationError
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from utils import *
import time

import json
def readRaymondCocoData(data_dir, use_depth=False):
	print('Reading dataset from ' + data_dir + 'dataset.json')
	json_file = open(data_dir + 'dataset.json', 'r')
	dataset = json.load(json_file)

	img_channels = 3
	if use_depth:
		img_channels = 4

	X = np.empty((len(dataset['annotations']), 32, 32, img_channels), dtype=np.uint8)
	y = np.empty((len(dataset['annotations']), 3), dtype=np.float)

	annotation_id = 0
	for img_data in dataset['images']:
		rgb_filename   = img_data['file_name']
		img_rgb   = cv2.imread(data_dir + rgb_filename)
		if use_depth:
			depth_filename = img_data["depth"]["file_name"]
			img_depth = cv2.imread(data_dir + depth_filename, cv2.IMREAD_ANYDEPTH)

		for annotation in dataset['annotations']:
			if annotation['image_id'] == img_data['id']:
				x_bbox, y_bbox, w_bbox, h_bbox = annotation['bbox']
				bbox = []
				if use_depth:
					bbox_rgb   = img_rgb[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
					bbox_depth = img_depth[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
					bbox = np.append(bbox_rgb, bbox_depth[..., np.newaxis], axis=-1)
				else:
					bbox = img_rgb[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
				bbox = cv2.resize(bbox, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

				X[annotation_id] = bbox
				y[annotation_id] = np.array(annotation['orientation'])
				annotation_id = annotation_id + 1

	X = X[:annotation_id,...]
	y = y[:annotation_id,...]
	return X, y


def createActivationMap(model, img, y_true):
	print(img.shape)
	img_tensor = preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)

	dense_weights = model.layers[-2].get_weights()[0]
	final_conv_layer = model.layers[-6]

	# Heatmap function with same input as model to output of final conv layer
	heatmap_model = models.Model([model.layers[0].input], [model.layers[-6].output, model.layers[-2].output])
	print(heatmap_model)
	[conv_outputs, predictions] = heatmap_model([img])
	conv_outputs = conv_outputs[0, :, :, :]

	np.degrees(np.arccos([np.dot(y_true[i], y_pred[i]) for i in range(len(y_true))]))


if __name__ == '__main__':
	setupGPU()

	use_color = True
	use_depth = True
	interpolation = cv2.INTER_LANCZOS4
	X01, y01 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/segmented_berries_test01/", use_color=use_color, use_depth=use_depth, interpolation=interpolation)
	X02, y02 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/segmented_berries_test02/", use_color=use_color, use_depth=use_depth, interpolation=interpolation)
	X03, y03 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/segmented_berries_test03/", use_color=use_color, use_depth=use_depth, interpolation=interpolation)
	X = np.concatenate((X01, X02), axis=0)
	X = np.concatenate((  X, X03), axis=0)
	y = np.concatenate((y01, y02), axis=0)
	y = np.concatenate((  y, y03), axis=0)
	#X, y = readRaymondCocoData("/home/niko/Downloads/riseholme_2019_pico_amesti/", use_depth=True)
	y_true = y

	model = load_model("orientation_estimator_segmented.hdf5", custom_objects={'RotationError': RotationError})
	t_start = time.time()
	y_pred = model.predict(X)
	t_end = time.time()
	print(t_end - t_start)

	y_true = normalize(y_true)
	y_pred = normalize(y_pred)

	angles = np.degrees(np.arccos([np.dot(y_true[i], y_pred[i]) for i in range(len(y_true))]))
#	for i, img in enumerate(X):
#		print("Sample #" + str(i + 1))
#		print("True: " + str(y_true[i]))
#		print("Pred: " + str(y_pred[i]))
#		print("Angle error: " + str(angles[i]) + "\n")

	results_str = ("Angular errors:         \n" +
	               "- Max error:    {:>7.3f}deg\n".format(np.max(angles)) +
	               "- Mean error:   {:>7.3f}deg\n".format(np.mean(angles)) +
	               "- Median error: {:>7.3f}deg\n".format(np.median(angles)))

	az_pred = np.degrees(np.arctan(y_true[:,1] / y_true[:,0]))
	el_pred = np.degrees(np.arctan(y_true[:,2] / y_true[:,0]))
	colors = cm.hot(angles / 180.0)


	# Load training data for reference plot
	train_orientations = np.load(open('train_orientations.npy', 'rb'))

	az_train = np.degrees(np.arctan(train_orientations[:,1] / train_orientations[:,0]))
	el_train = np.degrees(np.arctan(train_orientations[:,2] / train_orientations[:,0]))


	# Plot errors
	plt.rcParams.update({'font.size': 18})
	fig, (ax1, ax2) = plt.subplots(2)
	ax1.set_title('Absolute error distribution')
	ax1.set_xlabel('Angular error [deg]')
	ax1.set_ylabel('Number of samples')
	ax1.hist(angles, bins=20)
	plt.text(0.95, 0.95, results_str, transform = ax1.transAxes,
	         fontsize=18, fontdict={'family' : 'monospace'},
	         ha="right", va="top")
	plt.grid(color='lightgrey')
	ax2.set_title('Error distribution in relation to training data used')
	ax2.set_xlabel('Azimuth [deg]')
	ax2.set_ylabel('Elevation [deg]')
	ax2.scatter(az_train, el_train, s=3000, alpha=0.1, label='Training data')
	sc = ax2.scatter(az_pred, el_pred, c=(angles), cmap=cm.get_cmap("hot"), s=50, label='Prediction data')
	plt.colorbar(sc, label="Angular error [deg]")
	#ax2.legend(loc="upper right")
	plt.show()


	for i, angle in enumerate(angles):
		#if angle > 40:
		if angle < 120:
			#createActivationMap(model, X[i], y_true[i])
			print("Angular error: " + str(angle))
			print("- True: " + str(y_true[i]))
			print("- Pred: " + str(y_pred[i]))
			print("  -> Az: " + str(az_pred[i]))
			print("  -> El: " + str(el_pred[i]))
			print(img.shape)
			img              = cv2.resize(X[i], (512,512))
			arrow_startpoint = np.array((img.shape[0]/2, img.shape[1]/2))
			n                = np.array((1,0,0))
			dist             = np.dot(n, y_pred[i])
			arrow_endpoint   = (arrow_startpoint + (np.array((-1,-1)) * (y_pred[i] - dist * n)[1:3] * arrow_startpoint)).astype(int)

			print(arrow_startpoint)
			print(arrow_endpoint)
			#cv2.arrowedLine(img, tuple(arrow_startpoint), tuple(arrow_endpoint), (0, 255, 0), thickness=5)
			cv2.imshow("img", img[:,:,:3])
			cv2.imwrite("berry_" + str(i).zfill(3) + ".png", img[:,:,:3]) 
			#cv2.imshow("img_depth", X[i,:,:,3])
			cv2.waitKey()