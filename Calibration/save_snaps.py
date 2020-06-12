import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import numpy as np
import cv2
import requests

url = url = "http://[2401:4900:1ff9:83fa::da]:8080/shot.jpg"
image_folder = "//home//debz//Desktop//Work//Open CV learning//Modification//Augmented Reality//Calibration//Snaps//picture"
ctr = 0
max_ctr = 200
while(True):
    img_req = requests.get(url)
    img_arr = np.array(bytearray(img_req.content),dtype = np.uint8)
    image = cv2.imdecode(img_arr,-1)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(8,6),None)
    if ret:
        print(ctr)
        cv2.imwrite(str(image_folder) + str(ctr)+".jpg",image)
        ctr = ctr + 1
        image = cv2.drawChessboardCorners(image,(8,6),corners,ret)
    if ctr > max_ctr:
        break
    cv2.resize(image,(960,720))
    cv2.imshow("image",image)
    if cv2.waitKey(10)==27:
        break

cv2.destroyAllWindows()
