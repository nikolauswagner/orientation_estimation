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

import json

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
	X_rgbd = np.concatenate((X01, X02), axis=0)
	X_rgbd = np.concatenate((X_rgbd, X03), axis=0)

	use_color = True
	use_depth = False
	X01, y01 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/segmented_berries_test01/", use_color=use_color, use_depth=use_depth, interpolation=interpolation)
	X02, y02 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/segmented_berries_test02/", use_color=use_color, use_depth=use_depth, interpolation=interpolation)
	X03, y03 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/segmented_berries_test03/", use_color=use_color, use_depth=use_depth, interpolation=interpolation)
	X_rgb = np.concatenate((X01, X02), axis=0)
	X_rgb = np.concatenate((X_rgb, X03), axis=0)

	y = np.concatenate((y01, y02), axis=0)
	y = np.concatenate((  y, y03), axis=0)
	y_true = y

	model_rgbd = load_model("orientation_estimator_segmented.hdf5", custom_objects={'RotationError': RotationError})
	model_rgb  = load_model("orientation_estimator_nodepth.hdf5", custom_objects={'RotationError': RotationError})
	y_pred_rgbd = model_rgbd.predict(X_rgbd)
	y_pred_rgb  = model_rgb.predict(X_rgb)

	y_true = normalize(y_true)
	y_pred_rgbd = normalize(y_pred_rgbd)
	y_pred_rgb = normalize(y_pred_rgb)

	angles_rgbd = np.degrees(np.arccos([np.dot(y_true[i], y_pred_rgbd[i]) for i in range(len(y_true))]))
	angles_rgb  = np.degrees(np.arccos([np.dot(y_true[i], y_pred_rgb[i]) for i in range(len(y_true))]))

	results_rgbd_str = ("Angular errors (RGBD):   \n" +
	                    "- Max error:    {:>7.3f}deg\n".format(np.max(angles_rgbd)) +
	                    "- Mean error:   {:>7.3f}deg\n".format(np.mean(angles_rgbd)) +
	                    "- Median error: {:>7.3f}deg\n".format(np.median(angles_rgbd)))

	results_rgb_str = ("\n\n\n\n\nAngular errors (RGB):    \n" +
	                   "- Max error:    {:>7.3f}deg\n".format(np.max(angles_rgb)) +
	                   "- Mean error:   {:>7.3f}deg\n".format(np.mean(angles_rgb)) +
	                   "- Median error: {:>7.3f}deg\n".format(np.median(angles_rgb)))

	az_pred = np.degrees(np.arctan(y_true[:,1] / y_true[:,0]))
	el_pred = np.degrees(np.arctan(y_true[:,2] / y_true[:,0]))


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
	ax1.hist(angles_rgbd, bins=20, fc=(0, 0, 1, 0.3))
	ax1.hist(angles_rgb, bins=20, fc=(1, 0, 0, 0.3))
	plt.text(0.95, 0.95, results_rgbd_str, transform = ax1.transAxes,
	         fontsize=18, fontdict={'family' : 'monospace'},
	         ha="right", va="top", color='blue')
	plt.text(0.95, 0.95, results_rgb_str, transform = ax1.transAxes,
	         fontsize=18, fontdict={'family' : 'monospace'},
	         ha="right", va="top", color='red')
	plt.grid(color='lightgrey')
	ax2.set_title('Error distribution in relation to training data used')
	ax2.set_xlabel('Azimuth [deg]')
	ax2.set_ylabel('Elevation [deg]')
	ax2.scatter(az_train, el_train, s=3000, alpha=0.1, label='Training data')
	sc = ax2.scatter(az_pred, el_pred, c=(angles_rgbd), cmap=cm.get_cmap("hot"), s=50, label='Prediction data')
	plt.colorbar(sc, label="Angular error [deg]")
	#ax2.legend(loc="upper right")
	plt.show()