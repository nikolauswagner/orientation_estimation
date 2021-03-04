import matplotlib.pyplot as plt
import numpy as np
from utils import *

if __name__ == '__main__':

	X, y1 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/strawberries_01/", use_depth=False)
	X, y2 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/strawberries_02/", use_depth=False)
	X, y3 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/strawberries_03/", use_depth=False)
	X, y4 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/strawberries_04/", use_depth=False)
	X, y5 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/strawberries_05/", use_depth=False)
	X, y6 = readCocoLikeData("/home/niko/Documents/git/gazebo-data-annotator/annotated_data/strawberries_06/", use_depth=False)

	y = np.concatenate((y1, y2), axis=0)
	y = np.concatenate(( y, y3), axis=0)
	y = np.concatenate(( y, y4), axis=0)
	y = np.concatenate(( y, y5), axis=0)
	y = np.concatenate(( y, y6), axis=0)

	az = np.degrees(np.arctan(y[:,0] / y[:,1]))
	el = np.degrees(np.arctan(y[:,2] / y[:,1]))

fig, (ax1, ax2) = plt.subplots(2)
plt.grid(color='lightgrey')
ax2.set_title('Error distribution in relation to training data')
ax2.set_xlabel('Azimuth [deg]')
ax2.set_ylabel('Elevation [deg]')
ax2.scatter(az, el, s=1, label='Training data')
ax2.legend()
plt.show()