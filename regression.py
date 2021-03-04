import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import cosine_similarity
from keras import initializers

## TODO: change this to AngleError
#def AngleError(y_true, y_pred):
##	roll_err  = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(roll_true - roll_pred), 
##	                                           tf.cos(roll_true - roll_pred))))
##	pitch_err = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(pitch_true - pitch_pred), 
##	                                           tf.cos(pitch_true - pitch_pred))))
#
#	return tf.reduce_mean(tf.abs(tf.atan2(tf.sin(y_true - y_pred), 
#	                                      tf.cos(y_true - y_pred))))

# Requires y to be unit vectors
# Angular Distance (https://en.wikipedia.org/wiki/Cosine_similarity):
def RotationError(y_true, y_pred):
	y_true = K.l2_normalize(y_true, axis=-1)
	y_pred = K.l2_normalize(y_pred, axis=-1)
	return 1 + cosine_similarity(y_true, y_pred)
	#return 2 * tf.math.acos(K.sum(y_true * y_pred, axis=-1)) / np.pi

def PoseRegressor(input_shape, model_name='pose_regressor.hdf5'):
	initializer =  initializers.RandomNormal(mean=0., stddev=1.)

	# Essentially VGG16 with different output
	model = Sequential()
	model.add(Conv2D(filters=64,  kernel_size=(3,3), padding="same", activation="relu", input_shape=input_shape))
	model.add(Conv2D(filters=64,  kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Flatten())
	model.add(Dropout(0.1))
	model.add(Dense(3, activation="linear"))
	model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))  

	# Build model
	adam = Adam(lr = 0.001)
	model.compile(loss=RotationError, optimizer=adam)

	# Add callbacks
	checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=2, mode='auto')
	callbacks = [checkpoint, early_stopping]

	return model, callbacks


if __name__ == '__main__':
	model, callbacks = PoseRegressor((256,256,3))
	print(model.summary())