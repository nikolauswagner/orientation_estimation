import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == '__main__':
	roll  = - 60
	pitch = - 60
	yaw   = 0

	M_x = Rotation.from_euler('x', roll, degrees=True)
	M_y = Rotation.from_euler('y', pitch, degrees=True)
	M_z = Rotation.from_euler('z', yaw, degrees=True)
	M = M_x * M_y * M_z
	v = M.apply([0,0,1])
	
	print(v)

	angle = np.arccos(np.dot(v, np.array(([0,0,1]))))
	print(np.degrees(angle))