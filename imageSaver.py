import numpy as np
import cv2
import cv2 as cv
import math
import requests
import time
n=0
URL = "http://192.168.1.7:8080/shot.jpg"
while n<20:
    img = cv2.imread('shot.jpg',0)
    img_resp=requests.get(URL)
    img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
    #img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    cv2.imwrite('images/photo'+str(n)+'.jpg',img)
    cv2.waitKey(200)
    n+=1
