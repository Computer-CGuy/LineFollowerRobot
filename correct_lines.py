import numpy as np
import cv2
import cv2 as cv
img = 255-cv2.imread('TRACK1.png',0)
from matplotlib import pyplot as plt

#img = cv2.imread('image.pNg//')

kernel = np.ones((20,20), np.uint8)
img = cv2.erode(img, kernel, iterations=2)
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
    #cv2.imshow('img', img)
    #cv2.waitKey(100)
    if zeros==size:
        done = True
#cv2.imshow('Connt

skela = skel.copy()
edges = cv2.Canny(skela,50,150,apertureSize = 3)
cv2.imshow("edge",edges)
lines = cv2.HoughLines(edges,1,np.pi/180,100)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(skela,(x1,y1),(x2,y2),(222,0,255),20)
cv2.imshow('window',skela)
#cv2.imwrite('houghlines5.jpg',skel)

#cv2.imshow('window',skel)

