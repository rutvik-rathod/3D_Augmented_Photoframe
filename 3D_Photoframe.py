import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import requests

def getCameraMatrix():
    with np.load('/home/debz/Desktop/Work/Open CV learning/Modification/Augmented Reality/Calibration/Camera.npz') as X:
        camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    return camera_matrix, dist_coeff


#if you want to takein feed from web cam uncomment the below line
#cap = cv2.VideoCapture(0)

#The use of IP_WEBCAM_SERVER
url = "http://[2401:4900:1ff9:83fa::d1]:8080/shot.jpg"

project_image_path1 = "//home//debz//Desktop//Work//pictures//swag.jpg"
project_image_path2 = "//home//debz//Desktop//Work//pictures//ala.jpg"

#These are all the parameters responsible for video wrting
filename = '//home//debz//Desktop//Work//Open CV learning//hello.mp4'
codec = cv2.VideoWriter_fourcc('X','V','I','D')
framerate = 10
resolution = (960,720)
VideoFileOutput = cv2.VideoWriter(filename,codec,framerate,resolution)

#Video file Read
video = cv2.VideoCapture('//home//debz//Desktop//video_project.mp4')


#while using the webcam uncomment the below line and comment the line below
"""while(cap.isOpened()):
    ret,image = cap.read()
    if not ret:
        break
"""

cam,dist = getCameraMatrix()
while(True):
    img_req = requests.get(url)
    img_arr = np.array(bytearray(img_req.content),dtype = np.uint8)
    image = cv2.imdecode(img_arr,-1)
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
    parameters = aruco.DetectorParameters_create()
    corners,ids,_=aruco.detectMarkers(image,aruco_dict,parameters=parameters)
    if np.all(ids != None):

        markerLength = 100
        m = markerLength/2
        rvec,tvec,_ = aruco.estimatePoseSingleMarkers(corners,markerLength,cam,dist)
        image = aruco.drawDetectedMarkers(image,corners,ids) 
        pts = np.float32([[-m,m,0],[m,m,0],[m,-m,0],[-m,-m,0],
                          [-m,m,2*m],[m,m,2*m],[m,-m,2*m],[-m,-m,2*m]]) 
        
        pts_dict = {}
        pt_name = []
        imgpts,_ = cv2.projectPoints(pts,rvec,tvec,cam,dist)

        for i in range(8):
            pts_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())

        for i in range(8):
            pt_name.append(pts_dict[tuple(pts[i])])    

        
        project_image1 = cv2.imread(project_image_path1)
        project_image2 = cv2.imread(project_image_path2)
        
        size1 = project_image1.shape
        size2 = project_image2.shape

        pts_dist1 = np.array([pt_name[4],pt_name[7],pt_name[3],pt_name[0]])
        pts_dist2 = np.array([pt_name[5],pt_name[4],pt_name[0],pt_name[1]])
        
        pts_image1 = np.array([[0,0],
                               [size1[1]-1,0],
                               [size1[1]-1,size1[0]-1],
                               [0,size1[0]-1]], dtype = float);
        pts_image2 = np.array([[0,0],
                               [size2[1]-1,0],
                               [size2[1]-1,size2[0]-1],
                               [0,size2[0]-1]], dtype = float);
        
        im_dist = image

        h,status = cv2.findHomography(pts_image1,pts_dist1)
        temp = cv2.warpPerspective(project_image1,h,(im_dist.shape[1],im_dist.shape[0]))
        cv2.fillConvexPoly(im_dist,pts_dist1.astype(int),0,16)
        image = im_dist+temp

        h,status = cv2.findHomography(pts_image2,pts_dist2)
        temp = cv2.warpPerspective(project_image2,h,(im_dist.shape[1],im_dist.shape[0]))
        cv2.fillConvexPoly(image,pts_dist2.astype(int),0,16)
        image = image+temp

    
    image = cv2.resize(image,(960,720))
    VideoFileOutput.write(image)
    cv2.imshow('frame',image)
    k = cv2.waitKey(10)
    if(k==27):
        VideoFileOutput.release()
        break