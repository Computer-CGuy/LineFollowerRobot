import cv2
import numpy as np

img = 255-cv2.imread('TRACK.png',0)

kernel = np.ones((20,20), np.uint8)
img = cv2.erode(img, kernel, iterations=1)
cv2.imshow("lo",img)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False

while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy() 
    zeros = size - cv2.countNonZero(img)
    cv2.imshow('img', img)
    cv2.waitKey(100)
    if zeros==size:
        done = True
cv2.imshow("img",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
