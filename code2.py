import cv2
import numpy as np

img = cv2.imread('TRACK1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('houghlineas5.jpg',edges)
minLineLength = 10000
maxLineGap = 100
lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('houghlines5.jpg',img)
