from __future__ import division
import random
import numpy as np
from scipy import stats
import cv
import cv2
import copy as cp


class SystemModel:
    def __init__(self,model):
        self.model = model
    def generate(self,sv,w):
        return(self.model(sv,w))

def model_s(sv,w):
    F = np.matrix([[1,0,1,0,0,0,0],
                   [0,1,0,1,0,0,0],
                   [0,0,1,0,0,0,0],
                   [0,0,0,1,0,0,0],
                   [0,0,0,0,1,0,1],
                   [0,0,0,0,0,1,1],
                   [0,0,0,0,0,0,1]])
    return(np.array(np.dot(F,sv))[0]+w)

def getPixelPosition(rectangle,windows):
    rectanglepts = np.ones((rectangle.shape[0],rectangle.shape[1],1))*(1,1)
    for i in range(0,rectangle.shape[0],1):
        for j in range(0,rectangle.shape[1],1):
            rectanglepts[i,j] = (i+windows[0],j+windows[1])
    return rectanglepts


def calculateDistribution(window,rectangle,kmat,f,yuv,n):
    rectanglepts = getPixelPosition(rectangle,window)
    pY = 0
    for i in range(rectanglepts.shape[0]):
        for j in range(rectanglepts.shape[1]):
            tmp = rectangle[i][j][yuv]
            if (yuv>0):
                tmp=(tmp+1)/2
            #if (np.ceil(rectangle[i][j][0]*8)==n):
            if (np.ceil(rectangle[i][j][0]/32)==4):
                #pY = pY + kmat[i][j] * (np.ceil(tmp*8) - n + 1)*f
                pY = pY + kmat[i][j] * (np.ceil(tmp/32) - 4 + 1)*f
    return pY

def initStateVectors(imageSize,sampleSize):
    xs = [random.uniform(0,imageSize[1]) for i in range(sampleSize)]
    ys = [random.uniform(0,imageSize[0]) for i in range(sampleSize)]
    vxs = [random.uniform(0,5) for i in range(sampleSize)]
    vys = [random.uniform(0,5) for i in range(sampleSize)]
    hx = [1 for i in range(sampleSize)]
    hy = [1 for i in range(sampleSize)]
    a = [1 for i in range(sampleSize)]
    return([list(s) for s in zip(xs,ys,vxs,vys,hx,hy,a)])

def main():

    window = [100, 200, 350, 450] # y1, x1, y2, x2
    b = np.sqrt(np.power((window[2]-window[0]),2) + np.power((window[3]-window[1]),2))

    capture = cv2.VideoCapture('video/wei.avi')
    sampleSize = 100;
    imageSize = []
    imageSize.append(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    imageSize.append(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    ret, image = capture.read()
    yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV);
    #rectangle = image[window[0]:window[2],window[1]:window[3]]
    rectangle = yuv[window[0]:window[2],window[1]:window[3]]
    svs = initStateVectors(imageSize,sampleSize)

    dst = yuv.copy()
    for sv in svs:
        cv2.circle(dst,(int(sv[0]),int(sv[1])),3,cv.CV_RGB(100,0,255))

    # while (capture.isOpened()):
    #     ret, image = capture.read()
    #     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV);

    rectanglepts = getPixelPosition(rectangle,window)


    c = [(window[2]-window[0])/2,(window[3]-window[1])/2] # y, x
    cmat = np.ones((rectanglepts.shape[0],rectanglepts.shape[1],1))*c
    rmat = np.divide((rectanglepts - cmat),b)
    kmat = np.ones((rectangle.shape[0],rectangle.shape[1])) - rmat**2
    f = np.sum(1./kmat)
    distribution = np.zeros((3,8))
    for i in range(3):
        for j in range(8):
            distribution[i][j]=calculateDistribution(window,rectangle,kmat,f,i,j+1)

    cv2.imshow('frame1',dst)
    cv2.waitKey()

if __name__ == '__main__':
    main()