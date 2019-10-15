import numpy as np
import cv2
import cv2 as cv

ima = cv2.imread('track.jpg')
imgray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
im = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)






imm = cv2.inRange(im,(0),(49)) 

kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(imm, cv2.MORPH_GRADIENT, kernel)
il = cv2.dilate(gradient, kernel, iterations=7)
ol = cv2.erode(il, kernel, iterations=7)
cv2.imshow('aloc',ol)

contours,hei = cv2.findContours(ol,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(ima, contours, -1, (200,255,0), 3)
#for x in range(0,len(contours[1])-10,2):
#    a = ((contours[0][x+1]+contours[1][x+1]/2))[0]
#   b= ((contours[0][x+1]+contours[1][x+1]/2))[0]
#    cv2.line(ima,(int(a[1]),int(a[0])), (int(b[1]),int(b[0])), (0,0,200), 2)
    #print(x,y)
cnt = contours[0]

# then apply fitline() function
[vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)

# Now find two extreme points on the line to draw line
lefty = int((-x*vy/vx) + y)
righty = int(((im.shape[1]-x)*vy/vx)+y)

#Finally draw the line
cv2.line(ima,(im.shape[1]-1,righty),(0,lefty),255,2)
cv2.imshow('window',ima)
