import numpy as np
import os
import json
import cv2
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2

initialize_weights = "random_uniform"
initialize_bias    = "random_uniform"

def SiameseNetwork(input_shape):
	# Define the tensors for the two input images
	left_input  = Input(input_shape)
	right_input = Input(input_shape)

	# Convolutional Neural Network
	model = Sequential()
	model.add(Conv2D(filters=64, 
	                 kernel_size=(10,10), 
	                 activation='relu', 
	                 input_shape=input_shape,
  	               kernel_initializer=initialize_weights, 
	                 kernel_regularizer=l2(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(filters=128, 
	                 kernel_size=(7,7), 
	                 activation='relu',
	                 kernel_initializer=initialize_weights,
	                 bias_initializer=initialize_bias, 
	                 kernel_regularizer=l2(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(128, (4,4), 
	                 activation='relu', 
	                 kernel_initializer=initialize_weights,
	                 bias_initializer=initialize_bias, 
	                 kernel_regularizer=l2(2e-4)))
	model.add(MaxPooling2D())
	model.add(Conv2D(256, (4,4), 
	                 activation='relu', 
	                 kernel_initializer=initialize_weights,
	                 bias_initializer=initialize_bias, 
	                 kernel_regularizer=l2(2e-4)))
	model.add(Flatten())
	model.add(Dense(256, #4096
	                activation='sigmoid',
	                kernel_initializer=initialize_weights, 
	                bias_initializer=initialize_bias,
	                kernel_regularizer=l2(1e-3)))

	# Generate the encodings (feature vectors) for the two images
	encoded_l = model(left_input)
	encoded_r = model(right_input)

	# Add a customized layer to compute the absolute difference between the encodings
	L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
	L1_distance = L1_layer([encoded_l, encoded_r])

	# Add a dense layer with a sigmoid unit to generate the similarity score
	prediction = Dense(1,
	                   activation='sigmoid',
	                   bias_initializer=initialize_bias)(L1_distance)

	# Connect the inputs with the outputs
	siamese_net = Model(inputs=[left_input,right_input],
	                    outputs=prediction)

	# return the model
	return siamese_net



if __name__ == '__main__':
	model = SiameseNetwork((256, 256, 3))
	optimizer = Adam(lr = 0.00006)
	model.compile(loss="binary_crossentropy", optimizer=optimizer)

	img_dir = "/run/media/niko/2f66d643-9eda-48ff-9721-a567ef80ae6a/data/render/strawberry_01/"
	samples = {}
	samples["sets"] = []
	for i_set, img_set in enumerate(sorted(os.listdir(img_dir))):
		if i_set == 10:
			break # HACK: Only use 10 sets for the momemt
		print("Processing set #" + str(i_set))
		#input()
		samples["sets"].append([])
		for roll in range(0,360,10):
			for pitch in range(0,360,10):
				filename = img_dir + img_set + "/roll" + str(roll).zfill(3) + "_pitch" + str(pitch).zfill(3) + ".png"
				sample = {}
				sample["roll"]  = roll
				sample["pitch"] = pitch
				sample["rgb"]   = cv2.imread(filename)

				samples["sets"][i_set].append(sample)

	cv2.imshow("img",samples["sets"][5][1200]["rgb"])
	cv2.waitKey()