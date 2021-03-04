import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# Data for plotting
	x = np.arange(0.0, 180.0, 0.01)
	y1 = np.cos(np.radians(x))
	y2 = (1 - np.cos(np.radians(x))) / 2

#	fig, ax = plt.subplots(1, 2, figsize=(7, 4))
#	ax[0].plot(x, y1)
#	ax[0].set(xlabel='angle [deg]', ylabel='value',
#	       title='cosine similarity')
#	ax[0].grid()
#	ax[1].plot(x, y2)
#	ax[1].set(xlabel='angle [deg]',
#	       title='custom loss function')
#	ax[1].grid()

	fig, ax = plt.subplots(1, 1, figsize=(8, 4))
	ax.plot(x, y2)
	ax.set(xlabel='angle [deg]',
	       title='custom loss function')
	ax.grid()

	fig.savefig("cosine_loss.png")
	plt.show()