from regression import PoseRegressor
import numpy as np
from utils import setupGPU, readRenderData, readCocoLikeData
import cv2
import json
from scipy.spatial.transform import Rotation as Rotation

def readRealData(data_dir, interpolation=cv2.INTER_LANCZOS4):
	print('Reading dataset from ' + data_dir + 'training.json')
	json_file = open(data_dir + 'testing.json', 'r')
	dataset = json.load(json_file)

	X = np.empty((len(dataset['annotations']), 32, 32, 3), dtype=np.uint8)
	y = np.empty((len(dataset['annotations']), 3), dtype=np.float)

	annotation_id = 0
	for img_data in dataset['images']:
		if annotation_id == 5000: 
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

	print('Loading training data...')
	interpolation = cv2.INTER_LANCZOS4
	X, y = readRealData("/media/niko/2f66d643-9eda-48ff-9721-a567ef80ae6a/data/strawberries/dryad_geometric/", interpolation=interpolation)
	
	np.save(open('train_orientations_real.npy', 'wb'), y)

#	for i, img in enumerate(X):
#		img              = cv2.resize(img, (512,512))
#		arrow_startpoint = np.array((img.shape[0]/2, img.shape[1]/2))
#		n                = np.array((1,0,0))
#		dist             = np.dot(n, y[i])
#		arrow_endpoint   = (arrow_startpoint + (np.array((-1,-1)) * (y[i] - dist * n)[1:3] * arrow_startpoint)).astype(int)
#
#		cv2.arrowedLine(img, tuple(arrow_startpoint), tuple(arrow_endpoint), (0, 255, 0), thickness=5)
#		cv2.imshow("img", img[:,:,:3])
#		cv2.imwrite("berry_000.png", img[:,:,:3]) 
#		cv2.waitKey()

	model, callbacks = PoseRegressor(X[0].shape, model_name="orientation_estimator_real.hdf5")
	model.fit(X, y, 
	          batch_size=128, 
	          epochs=100, 
	          verbose=1, 
	          validation_split=0.1,
	          callbacks=callbacks)