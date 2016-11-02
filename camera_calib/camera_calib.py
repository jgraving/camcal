import numpy as np
import cv2
import pickle

def calibrate_camera_still(calib_images,  grid_size = (9,6), imshow = True, delay = 500):

	""" Returns calibration parameters from calibration images

		Parameters
		----------
		calib_images : list of strings
			List of file paths as strings (e.g. glob.glob("/path/to/files/*.jpg").
		grid_size : tuple of int
			Size of calibration grid (internal corners)
		imshow : bool
			Show the calibration images
		delay : int, >=1
			Delay in msecs between each image for imshow
			
		Returns
		-------
		calib_params : dict
			Parameters for undistorting images.

	"""

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((grid_size[1]*grid_size[0],3), np.float32)
	objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	for filename in calib_images:
		img = cv2.imread(filename)

		if img != None:

			#img = cv2.resize(img, (0,0), None, fx = 0.25, fy = 0.25)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, grid_size, None, flags = (cv2.CALIB_CB_ADAPTIVE_THRESH))
			#ret, corners = cv2.findCirclesGrid(gray, grid_size, None, flags = (cv2.CALIB_CB_ASYMMETRIC_GRID))
			
			# If found, add object points, image points (after refining them)
			if ret == True:
				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners2)

				if imshow == True:
					img = cv2.drawChessboardCorners(img, grid_size, corners2, ret)

			# Draw and display the corners
			if imshow == True:
				cv2.imshow('img',img)
				cv2.waitKey(delay)
				
	if imshow == True:
		cv2.destroyAllWindows()
		for i in range(5):
			cv2.waitKey(1) 

	if len(objpoints) > 0 and len(imgpoints) > 0:
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		calib_params = {"ret" : ret, "mtx" : mtx, "dist" : dist, "rvecs" : rvecs, "tvecs" : tvecs}
		
		total_error = 0
		for i in xrange(len(objpoints)):
			imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			total_error += error 
		mean_error = total_error/len(objpoints)

		print "Calibration successful! Mean error: ", mean_error

	else:
		print "No calibration points found!"
		calib_params = None

	return calib_params


def undistort_image(image, calib_params, crop = True):

	""" Returns undistorted image using calibration parameters.

		Parameters
		----------
		image : numpy_array 
			Image to be undistorted
		calib_params : dict
			Calibration parameters from calibrate_camera()
		crop : bool
			Crop the image to the optimal region of interest
			
		Returns
		-------
		dst : numpy_array
			Undistorted image.

	"""
	try:
		ret = calib_params["ret"]
		mtx = calib_params["mtx"]
		dist = calib_params["dist"]
		rvecs = calib_params["rvecs"]
		tvecs = calib_params["tvecs"]
	except:
		raise TypeError("calib_params must be 'dict'")

	img = image
	h,  w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

	# undistort
	mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
	dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


	# crop the image
	if crop:
		x,y,w,h = roi
		dst = dst[y:y+h, x:x+w]

	return dst

def save_calib(filename, calib_params): 
	""" Saves calibration parameters as '.pkl' file.

		Parameters
		----------
		filename : str
			Path to save file, must be '.pkl' extension
		calib_params : dict
			Calibration parameters to save

		Returns
		-------
		saved : bool
			Saved successfully.
	"""
	if type(calib_params) != dict:
			raise TypeError("calib_params must be 'dict'")

	output = open(filename, 'wb')

	try:
		pickle.dump(calib_params, output)
	except:
		raise IOError("filename must be '.pkl' extension")

	output.close()

	saved = True

	return saved

def load_calib(filename):
	""" Loads calibration parameters from '.pkl' file.

		Parameters
		----------
		filename : str
			Path to load file, must be '.pkl' extension
			
		Returns
		-------
		calib_params : dict
			Parameters for undistorting images.

	"""
	# read python dict back from the file
	
	pkl_file = open(filename, 'rb')

	try:
		calib_params = pickle.load(pkl_file)
	except:
		raise IOError("File must be '.pkl' extension")

	pkl_file.close()

	return calib_params