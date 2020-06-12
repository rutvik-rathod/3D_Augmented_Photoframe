import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import numpy as np
import cv2
import requests

image_folder = "//home//debz//Desktop//Work//Open CV learning//Modification//Augmented Reality//Calibration//Snaps//picture"
npz_path = "/home/debz/Desktop/Work/Open CV learning/Modification/Augmented Reality/Calibration/Camera.npz"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8,3),np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
# print(objp)

objpoints = []
imgpoints = []

ctr = 0
for ctr in range(201):
    image = cv2.imread(str(image_folder) + str(ctr) +".jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,(8,6),None)

    if ret:
        print(ctr)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1], None, None)
print(mtx,dist)
np.savez(npz_path, dist=dist, ret=ret, mtx=mtx, rvecs=rvecs,tvecs=tvecs)
