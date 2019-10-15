import cv2
import numpy as np

# read image and invert so blob is white on black background
img = 255-cv2.imread('TRACK1.png',0)

# do some eroding of img, but not too much
kernel = np.ones((20,20), np.uint8)
img = cv2.erode(img, kernel, iterations=2)

# threshold img
ret, thresh = cv2.threshold(img,127,255,0)

# do distance transform
dist = cv2.distanceTransform(thresh, distanceType=cv2.DIST_L2, maskSize=5)

# set up cross for tophat skeletonization
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
skeleton = cv2.morphologyEx(dist, cv2.MORPH_TOPHAT, kernel)

# threshold skeleton
ret, skeleton = cv2.threshold(skeleton,0,255,0)

cv2.imshow("skeleton",skeleton)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)


# display skeleton
cv2.imshow("skeleton",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save results
cv2.imwrite('tall_blob_skeleton.png', skeleton)
