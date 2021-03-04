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
from scipy.spatial.transform import Rotation as Rotation
import json
import time

def readRealData(data_dir, interpolation=cv2.INTER_LANCZOS4):
	print('Reading dataset from ' + data_dir + 'testing.json')
	json_file = open(data_dir + 'testing.json', 'r')
	dataset = json.load(json_file)

	X = np.empty((len(dataset['annotations']), 32, 32, 3), dtype=np.uint8)
	y = np.empty((len(dataset['annotations']), 3), dtype=np.float)

	annotation_id = 0
	for img_data in dataset['images']:
		if annotation_id == 200: 
			break

		rgb_filename = img_data['file_name']
		# Evil data with different orientations
		if "SanAndreas" in rgb_filename:
			continue
		img_rgb = cv2.imread(data_dir + rgb_filename)

		for annotation in dataset['annotations']:
			if annotation['image_id'] == img_data['id']:
				x_bbox, y_bbox, w_bbox, h_bbox = annotation['bbox']
				bbox = img_rgb[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]	
				#cv2.imshow('img', bbox)
				bbox = cv2.resize(bbox, dsize=(32, 32), interpolation=interpolation)
				X[annotation_id] = bbox
				capture_num = int(rgb_filename[rgb_filename.rfind("_")+1:-4])
				if capture_num <= 11:
					v = np.array((1,0,0))
					rot1 = Rotation.from_euler('z', capture_num * 18, degrees=True)
					rot2 = Rotation.from_euler('x', -10, degrees=True)
					y[annotation_id] = rot2.apply(rot1.apply(v))
				else:
					v = np.array((0,0,1))
					rot = Rotation.from_euler('x', (capture_num - 13) * 18, degrees=True)
					y[annotation_id] = rot.apply(v)

				annotation_id = annotation_id + 1

		#print(rgb_filename)
		#print(y[annotation_id-1])	
		#cv2.waitKey()

	X = X[:annotation_id,...]
	y = y[:annotation_id,...]
	return X, y


if __name__ == '__main__':
	setupGPU()

	interpolation = cv2.INTER_LANCZOS4
	X, y = readRealData("/media/niko/2f66d643-9eda-48ff-9721-a567ef80ae6a/data/strawberries/dryad_geometric/", interpolation=interpolation)
	y_true = y

	model = load_model("orientation_estimator_real.hdf5", custom_objects={'RotationError': RotationError})
	t_start = time.time()
	y_pred = model.predict(X)
	t_end = time.time()
	print(t_end - t_start)

	y_true = normalize(y_true)
	y_pred = normalize(y_pred)

	angles = np.degrees(np.arccos([np.dot(y_true[i], y_pred[i]) for i in range(len(y_true))]))
	#for i, img in enumerate(X):
	#	print("Sample #" + str(i + 1))
	#	print("True: " + str(y_true[i]))
	#	print("Pred: " + str(y_pred[i]))
	#	print("Angle error: " + str(angles[i]) + "\n")

	results_str = ("Angular errors:         \n" +
	               "- Max error:    {:>7.3f}deg\n".format(np.max(angles)) +
	               "- Mean error:   {:>7.3f}deg\n".format(np.mean(angles)) +
	               "- Median error: {:>7.3f}deg\n".format(np.median(angles)))

	az_pred = np.degrees(np.arctan(y_true[:,1] / y_true[:,0]))
	el_pred = np.degrees(np.arctan(y_true[:,2] / y_true[:,0]))
	colors = cm.hot(angles / 180.0)


	# Load training data for reference plot
	train_orientations = np.load(open('train_orientations_real.npy', 'rb'))

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
		#if angle > 30:
		if angle < 10:
			#createActivationMap(model, X[i], y_true[i])
			print("Angular error: " + str(angle))
			print("- True: " + str(y_true[i]))
			print("- Pred: " + str(y_pred[i]))
			print("  -> Az: " + str(az_pred[i]))
			print("  -> El: " + str(el_pred[i]))
			img              = cv2.resize(X[i], (512,512))
			arrow_startpoint = np.array((img.shape[0]/2, img.shape[1]/2))
			n                = np.array((1,0,0))
			dist             = np.dot(n, y_pred[i])
			arrow_endpoint   = (arrow_startpoint + (np.array((-1,-1)) * (y_pred[i] - dist * n)[1:3] * arrow_startpoint)).astype(int)

			print(arrow_startpoint)
			print(arrow_endpoint)
			cv2.arrowedLine(img, tuple(arrow_startpoint), tuple(arrow_endpoint), (0, 255, 0), thickness=5)
			cv2.imshow("img", img[:,:,:3])
			cv2.imwrite("berry_" + str(i).zfill(3) + ".png", img[:,:,:3]) 
			cv2.waitKey()