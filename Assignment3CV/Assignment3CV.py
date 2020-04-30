import time
#Best Version
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import solve

image1 = cv2.imread('E:/TERM 9/Computer Vision/Assignment3CV/image4_1.jpg', 3)
image2 = cv2.imread('E:/TERM 9/Computer Vision/Assignment3CV/image4_2.jpg', 3)
cv2.imshow('First Image', image1)
cv2.imshow('Second Image', image2)
cv2.waitKey(0) 
cv2.destroyAllWindows()



def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()
    
plt.imshow(image1)
plt.setp(plt.gca(), autoscale_on=False)

tellme('click to begin')

plt.waitforbuttonpress()

ptsim1 = []
tellme('Select n points with mouse')
ptsim1 = np.asarray(plt.ginput(10, timeout=-1))
print(ptsim1)
print(ptsim1[0][0])
print(ptsim1[0][1])
plt.imshow(image2)
plt.setp(plt.gca(), autoscale_on=False) #gca= get current axis #setp sets properties

tellme('click to begin')

plt.waitforbuttonpress()

ptsim2 = []
tellme('Select n points with mouse')
ptsim2 = np.asarray(plt.ginput(10, timeout=-1))
print(ptsim2)

#h, status = cv2.findHomography(ptsim1, ptsim2)
def getrows(xi, yi, xpi, ypi):
    print(xi)
    pi = np.matrix([[xi, yi, 1, 0, 0, 0, -xi * xpi, -yi * xpi], [0, 0, 0, xi, yi, 1, -xi * ypi, -yi * ypi]])

    return pi


#print(h)


def getP(arrayp1, arrayp2, nclicks):
    b = []
    myp1 = []
    myp1 = np.concatenate([getrows(arrayp1[s][0], arrayp1[s][1], arrayp2[s][0], arrayp2[s][1]) for s in range(0, nclicks)])
    for q in range(0, nclicks):
        b.append([arrayp2[q][0]])
        b.append([arrayp2[q][1]])
    print(myp1.shape)
    b = np.asarray(b)
    print("de bbbbb")
    print(b)
    return myp1, b

#LSTSQ SOLUTION NON HOMOGENOUS AX=B 

A, b = getP(ptsim1, ptsim2, 10)
h = np.linalg.lstsq(A, b)
h = h[0]
# print(H[0])
print('printaya fl nos fadya')

H = []
for elem in h:
    H.append(elem[0])
H.append(1)
H = np.asarray(H).reshape((3, 3))
print(H)

# PH=0    
#Zero=np.matrix([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])
#HAN=np.matrix([[],[],[],[],[],[],[],[],[1]])


#HOMOGENEOUS PH=0 SVD SOLUTION
#A=getP(ptsim1,ptsim2,5)

#u, s, v = np.linalg.svd(A)

#hHomo = np.reshape(v[8], (3, 3))
#print(hHomo)
#print(hHomo.item(8))
#hHomo=np.dot(1/h.item(8),hHomo)
#print(hHomo)


h2, status = cv2.findHomography(ptsim1, ptsim2)
print('Built-in Function Result')
print(h2)

#H6 = [[0.585512,0.128589,259.396],[-0.285436,0.86002,52.1516],[-0.000748086,-0.000101119,1]] #Mountains
#H8 = [[7.65394952e-01,3.76570329e-02,4.46518040e+02],[-1.35506629e-01,9.12045564e-01,7.60749523e+01],[-2.11142201e-04,-3.21686931e-05,1.00000000e+00]] #Towers
#VERIFICATION PART
#fig = plt.figure()
#figB = fig.add_subplot(1,2,2)
#figA = fig.add_subplot(1,2,1)
#figB.imshow(image2,origin='upper')
#figA.imshow(image1,origin='upper')
#plt.axis('image')
#i = 0
#while (i < (20/2)):
    #pts = plt.ginput(1,timeout=0)
    #figA.scatter([pts[0][0]],[pts[0][1]])
    #pts = np.reshape(pts,(1*2,1))
    #toTrans = np.ones((3,1))
    #print(toTrans)
    #print(pts)
    #toTrans[0][0] = pts[0]
    #toTrans[1][0] = pts[1]
    #print(toTrans)
    #p = np.dot(H6,toTrans)
    #print(p)
    #print(p[0][0],p[2,0])
    #x = p[0][0]/p[2,0]
    #x=x-image2.shape[1]-180
    #y = p[1][0]/p[2,0]
    #y = y+85
    #figB.scatter([x],[y])
    #i = i + 1
#def WarpingFunction(H,Image):
#H = [[0.585512,0.128589,259.396],[-0.285436,0.86002,52.1516],[-0.000748086,-0.000101119,1]]

#RANSAC BUILT-IN
sift= cv2.xfeatures2d.SIFT_create()

#Find key points and descriptors
keypoint1,desc1=sift.detectAndCompute(image1,None)
keypoin2,desc2=sift.detectAndCompute(image2,None)
bf= cv2.BFMatcher() #brute-force matcher
matches=bf.knnMatch(desc1,desc2,k=2) #matches between descriptors of 2 images 
matches=np.asarray(matches) #Matches found
print('here')
if (len(matches[:,0]) >= 4):
    srcview= np.float32([keypoint1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2) #query is index for keypoints1, trainIdx is index for kp2
    dstview= np.float32([keypoint2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
H3,masked = cv2.findHomography(srcview,dstview,RANSAC,5.0) #5 threshold allowed reprojection error to treat a point as an outlier.
print('Automatic Homography')
print(H3)


print('WARPING')
img2 = cv2.imread('E:/TERM 9/Computer Vision/Assignment3CV/image6_1.jpg', 3)
img1 = cv2.imread('E:/TERM 9/Computer Vision/Assignment3CV/image6_2.jpg', 3)
mv1 = []
mv2 = []
rows = img1.shape[0]
cols = img2.shape[1] + img2.shape[1]
results = [np.zeros((rows,cols),np.uint8),np.zeros((rows,cols),np.uint8),np.zeros((rows,cols),np.uint8)]
mv1 = cv2.split(img1,mv1)
mv2 = cv2.split(img2,mv2)
print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')

for ii in range(0,3):
    img1 =mv1[ii]
    img2 =mv2[ii]
    
    
    Hinv = np.linalg.inv(h2)
    pixel = np.ones((3,1))
    transPix = np.zeros((3,1),np.float64)    
    
    for i in range(0,img1.shape[0]):    #loop on y
            for j in range(0,img1.shape[1]):    #loop on x
                pixel[0][0] = j
                pixel[1][0] = i
                pixel[2][0] = 1  
                
                transPix = np.dot(h2,pixel)
                x = transPix[0][0] / transPix[2][0]
                #x = x+img2.shape[1]
                y = transPix[1][0] / transPix[2][0]
                l = math.floor(x)
                k = math.floor(y)
                
                if(k< results[ii].shape[0]and l < results[ii].shape[1] and k >=0 and l>=0):
                    results[ii][k][l] = img1[i][j]
                    #fill holes using inverse wrapping
                    invWrap = np.zeros((3,1),np.float64)
                    uprow = np.int(k-1)
                    leftcol = np.int(l-1)
                    downrow = np.int(k+1)
                    rightcol = np.int(l+1)
                    for r in range(uprow,downrow):
                        for c in range(leftcol,rightcol):
                            if (r == k and c == l):
                                continue
                            if(r>0 and r <results[ii].shape[0] and c > 0 and c < results[ii].shape[1]):
                                invWrap[0][0] = c
                                #-img2.shape[1]
                                invWrap[1][0] = r
                                invWrap[2][0] = 1
                                invWrap = np.dot(Hinv,invWrap)
                                x = invWrap[0][0] / invWrap[2][0]
                                y = invWrap[1][0] / invWrap[2][0]
                                #print(x)
                                #print(y)
                                                            
                                if(math.floor(x) < img1.shape[1] and math.floor(x)>0 and math.floor(y) < img1.shape[0] and math.floor(y)>0):
                                    results[ii][r][c] = img1[math.floor(y)][math.floor(x)]
                
                
           
    for i in range(0,img2.shape[0]):
        for j in range(0,img2.shape[1]):
            results[ii][i][j] = img2[i][j]
    print ("channel 5elset")

    
#Concatenating All 3 channels
output = cv2.merge(results)
cv2.imshow("Final Output",output)
cv2.waitKey(0)    