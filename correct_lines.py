import numpy as np
import cv2
import cv2 as cv
import math
import requests
import time
#from matplotlib import pyplot as plt
def calc_c(theta):

    # theta = l/r
    r = 16 #cm
    # 3.5 cm Radius of whell
    length = math.radians(theta)*r
    return (length/2*math.pi*(3.5))
def power(theta):

    l = calc_c(theta)
    print(l)
    leftMotor = 255
    rightMotor = 255
    if(l>=0):
        leftMotor = (l-1)*255
    else:
        l = l*-1
        rightMotor = (1-l)*255
    return [leftMotor,rightMotor]
def roi(img, vertices):
    #blank mask:
    #print(vertices)
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked
def decode(thetha1,thetha2):
    #thetha= thetha2
    thetha = thetha1 + ((90-abs(thetha1))/90)*thetha2
    #thetha = 180+((thetha1-90)/90)*thetha2
    return thetha
def calibrate(follower):
    return np.linalg.solve([[follower[0],follower[1]], [imaginary[0],imaginary[1]]],[follower[2],imaginary[2]])
def w(n, d):
    return n / d if d else 0
def GetAngle (p1, p2,state):
    #state=0
    x1, y1 = p1
    x2, y2 = p2
    m1 = w(x1,y1)
    m2 = w(x2,y2)
    tnAngle = abs((m1-m2)/(1+(m1*m2)))
    tnAngle = math.degrees(math.atan(tnAngle))
    tnAngle=90-tnAngle
    #print(str(m1)+" "+str(m2))
    if(state==2):
        if(m1<0):
            tnAngle*=-1
    if(state==1):
        if(m2<0):
            tnAngle*=-1
    return tnAngle
def lineFromPoints(P,Q):

    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a*(P[0]) + b*(P[1])

    return [a, b,c]
URL = "http://192.168.1.7:8080/shot.jpg"
n = 0
bota = [0,0]

while True:
    img = cv2.imread('shot.jpg',0)
    img_resp=requests.get(URL)
    img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
    #img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)

    #img = cv2.inRange(img, (255,255,255), (60,60,60))
    #cv2.imshow("A",img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.waitKey(10)"""
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    pts = np.array([[0,img.shape[1]],[0,img.shape[1]*1/5],[img.shape[0]*1/5,img.shape[1]*1/10],[img.shape[0]*4/5,img.shape[1]*1/10],[img.shape[0],img.shape[1]*1/4],[img.shape[0],img.shape[1]]])
    img = roi(img,np.int32([pts]))
    #cv2.imshow("Camera",img)
    #cv2.waitKey(1)
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #cv2.imshow("TH3",i
    img = 255-img

    #cv2.waitKey(10)
    #img = 255-cv2.imread('TRACK1.png',0)


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
        #cv2.waitKey(1)
        if zeros==size:
            done = True
    #cv2.imshow('Connt',skel)
    #cv2.waitKey(10)
    skela = skel.copy()
    edges = cv2.Canny(skela,50,150,apertureSize = 3)
    #cv2.imshow("edge",edges)
    lines = cv2.HoughLines(edges,1,np.pi/180,80)
    #cv2.imshow("LINES_E",edges)

    for rho,theta in lines[len(lines)-1]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(222,0,255),3)
        follower = lineFromPoints([x1,y1],[x2,y2])
        break

    #cv2.imshow("LINES",img)
    cv2.waitKey(10)
    x,y = img.shape
    y -= 200

    imaginary = lineFromPoints([0,int(y*(3/4))],[x,int(y*(3/4))])
    if(n==0):
        loca = np.linalg.solve([[follower[0],follower[1]], [imaginary[0],imaginary[1]]],[follower[2],imaginary[2]])
        loc = loca[0]
        n+=1
    target = (int(loc),int(y*(3/4)))



    real = np.linalg.solve([[follower[0],follower[1]], [imaginary[0],imaginary[1]]],[follower[2],imaginary[2]])

    #print(real)


    real = np.linalg.solve([[follower[0],follower[1]], [imaginary[0],imaginary[1]]],[follower[2],imaginary[2]])
    #print(real)
    bot = (int(loc),int(y*(3.3/4)))
    line1 = lineFromPoints([int(loc),int(y*(3/4))],[int(loc),int(y*(3.3/4))])
    line2 = lineFromPoints([int(real[0]),int(real[1])],[int(loc),int(y*(3.3/4))])
    thetha1=(GetAngle((line1[0],line1[1]),(line2[0],line2[1]),1))
    thetha2=(GetAngle((follower[0],follower[1]),(imaginary[0],imaginary[1]),2))
    if(n==0):
        sub = thetha2

    #if(thetha2>90):
    #   thetha = (((thetha1+90)/90)*thetha2)-180
       #thetha*=-1
    if(thetha1==90):
        thetha1=0

    thetha = decode(thetha1,thetha2)
    print(thetha)
    print(power(thetha))
    cv2.line(img,(int(real[0]),int(real[1])),(int(loc),int(y*(3.3/4))),(255,255,255),2)
    cv2.line(img,bot,target,(255,255,255),2)
    img = cv2.circle(img, bot , 4, (255,255,255), 10)
    img = cv2.circle(img, target , 2, (255,255,255), 2)
    img = cv2.circle(img, (int(real[0]),int(real[1])) , 2, (255,255,255), 10)
    #cv2.imwrite('houghlines5.jpg',skel)
    cv2.line(img,(0,int(y*(3/4))),(x,int(y*(3/4))),(255,255,255),2)
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('window',img)



    #print(type(exception).__name__)
    pass

    #cv2.imshow('window',skel)
