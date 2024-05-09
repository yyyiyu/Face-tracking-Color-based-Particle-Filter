from __future__ import division
from scipy import stats
import random
import numpy as np
import cv
import cv2
import numpy as np
import time
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

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


def getWindow(pt,windowsize,frame):
    pt = np.asarray(pt)
    xmax = frame.shape[1]
    ymax = frame.shape[0]
    hx = int(windowsize[1]/2)
    hy = int(windowsize[0]/2)
    y = int(pt[1].round())
    x = int(pt[0].round())
    if ((y-hy>0) & (x-hx>0) & (y+hy<ymax) & (x+hx<xmax)):
        window = [y-hy,x-hx,y+hy,x+hx]
    else :
        window = -1
    return window

def getRectangle(frame,window):
    print(window)
    yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV);
    rectangle = yuv[window[0]:window[2],window[1]:window[3]]
    return rectangle

def calculateDistribution(window,rectangle,yuv,n,kmat,f,b):
    #rectanglepts = getPixelPosition(rectangle,window)
    pY = 0
    # cy = (window[2]+window[0])/2
    # cx = (window[3]+window[1])/2
    # f = 0

    for i in range(rectangle.shape[0]):
        for j in range(rectangle.shape[1]):
            tmp = rectangle[i][j][yuv]
            # r = np.sqrt(np.power((i+window[0]-cy),2)+np.power((j+window[1]-cx),2))/b
            # k = 1 - np.power(r,2)
            # f = f + k
            if (np.ceil(tmp/32)==n):
                pY = pY + kmat[i][j] * (np.ceil(tmp/32) - n + 1)*f
                # pY = pY + k*(np.ceil(tmp/32) - n + 1)
    # pY = pY/f
    return pY

def getDistributionMatrice(window,rectangle,kmat,f,b):
    distribution = np.zeros((3,8))
    for i in range(3):
        for j in range(8):
            distribution[i][j]=calculateDistribution(window,rectangle,i,j+1,kmat,f,b)
    return distribution


def getPixelPosition(rectangle,windows):
    rectanglepts = np.ones((rectangle.shape[0],rectangle.shape[1],1))*(1,1)
    for i in range(0,rectangle.shape[0],1):
        for j in range(0,rectangle.shape[1],1):
            rectanglepts[i,j] = (i+windows[0],j+windows[1])
    return rectanglepts


def getKmat(rectangle,window,b):
    rectanglepts = getPixelPosition(rectangle,window)
    c = [(window[2]+window[0])/2,(window[3]+window[1])/2] # y, x
    cmat = np.ones((rectanglepts.shape[0],rectanglepts.shape[1],1))*c
    rmat = np.divide(distancesPts(rectanglepts,cmat),b)
    kmat = np.ones((rectangle.shape[0],rectangle.shape[1])) - rmat**2
    return kmat

def initStateVectors(imageSize,sampleSize):
    xs = [random.uniform(0,imageSize[1]) for i in range(sampleSize)]
    ys = [random.uniform(0,imageSize[0]) for i in range(sampleSize)]
    vxs = [random.uniform(0,5) for i in range(sampleSize)]
    vys = [random.uniform(0,5) for i in range(sampleSize)]
    hx = [1 for i in range(sampleSize)]
    hy = [1 for i in range(sampleSize)]
    a = [1 for i in range(sampleSize)]
    return([list(s) for s in zip(xs,ys,vxs,vys,hx,hy,a)])

def distancesPts(mat1,mat2):
    distances = np.zeros((mat1.shape[0],mat1.shape[1]))
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            distances[i][j] = np.sqrt(np.power((mat1[i][j][0]-mat2[i][j][0]),2)+np.power((mat1[i][j][1]-mat2[i][j][1]),2))
    return distances

def draw_particles(svs_predict,frame):
    for sv in svs_predict:
        cv2.circle(frame,(int(sv[0]),int(sv[1])),3,(100,0,255))
    return frame

def getDistance(distribution,distributionNew):
    d = np.zeros((1,3))
    for i in range(3):
        ro = 0
        for j in range(8):
            ro = ro + np.sqrt(distribution[i][j]*distributionNew[i][j])
        d[0][i] = np.sqrt(1-ro)
    return d

def calculWeight(d,sigma):
    dprime = d.transpose()
    sigmadet = np.linalg.det(sigma)
    sigmainv = np.linalg.inv(sigma)
    weight = np.exp((-1/2)*(np.dot((np.dot(d,sigmainv)),dprime))) / (np.sqrt(np.power(2*np.pi,3)*sigmadet))
    return weight[0]

def resampling(svs,weights,N):
    sorted_particle = sorted([list(x) for x in zip(svs,weights)],key=lambda x:x[1],reverse=True)
    resampled_particle = []
    while(len(resampled_particle)<N):
        for sp in sorted_particle:
            resampled_particle += [sp[0]]*(np.array(sp[1])[0]*N*4)
    resampled_particle = resampled_particle[0:N]

    return(resampled_particle)

def addBruit(systemModel,sampleSize,dim,svs_predict):
    sigma = 10
    rnorm = stats.norm.rvs(0,sigma,size=sampleSize*dim)
    ranges = zip([sampleSize*i for i in range(dim)],[sampleSize*i for i in (range(dim+1)[1:])])
    ws = np.array([rnorm[p:q] for p,q in ranges])
    ws = ws.transpose()
    svs_predictNew = [systemModel.generate(sv,w) for sv,w in zip(svs_predict,ws)]
    return svs_predictNew

def main():

    # Initial the window that contains the object to be traced, for the input video, we track the face of a climber
    window = [80, 210, 160, 290] # y1, x1, y2, x2
    windowsize = [window[2]-window[0],window[3]-window[1]] #Hy, Hx
    logging.info(f"Initialized object window size H#{windowsize[0]}, W#{windowsize[1]}")

    b = np.sqrt(np.power((window[2]-window[0]),2) + np.power((window[3]-window[1]),2))
    sigmaweight = np.array([[1,0,0],[0,1,0],[0,0,1]])

    # Define output video file
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter('output/yL1uOzsjXlo_000003_000013_ft.avi',fourcc, 20, (480,360))

    # Load input video
    capture = cv2.VideoCapture('video/yL1uOzsjXlo_000003_000013.avi')

    # Initial numbre of particles used to prediction the next position of tracked object 
    sampleSize = 30

    # Initial rectangle
    imageSize=[capture.get(cv2.CAP_PROP_FRAME_HEIGHT),capture.get(cv2.CAP_PROP_FRAME_WIDTH)]
    logging.info(f"Input video frame size H#{imageSize[0]}, W#{imageSize[1]}")
    ret, image = capture.read()
    logging.info(f"capture read image succeed: {ret}")
)
    svs = initStateVectors(imageSize,sampleSize)
    #Draw particle on frame and show frame
    frameInit = draw_particles(svs,image)
    cv2.imshow('frame0',frameInit)
    time.sleep(5)
    capture.release()
    exit(0)
    yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV);
    rectangle = yuv[window[0]:window[2],window[1]:window[3]]
    #Calculate color distribution for the target rectangle
    kmat = getKmat(rectangle,window,b)
    f = 1/(np.sum(kmat))
    distribution = getDistributionMatrice(window,rectangle,kmat,f,b)

    dim = len(svs[1])
    # sigma = 2.0
    testcount=10
    while(capture.isOpened() and testcount>0):
        ret, frame = capture.read()
        logging.info(f"capture read image succeed: {ret}")

        systemModel = SystemModel(model_s)
        # rnorm = stats.norm.rvs(0,sigma,size=sampleSize*dim)
        # ranges = zip([sampleSize*i for i in range(dim)],[sampleSize*i for i in (range(dim+1)[1:])])
        # ws = np.array([rnorm[p:q] for p,q in ranges])
        # ws = ws.transpose()
        # svs_predict = [systemModel.generate(sv,w) for sv,w in zip(svs,ws)]
        svs_predict = svs #addBruit(systemModel,sampleSize,dim,svs)
        weights = []
        particlesValide = []
        for particle in svs_predict:
            windowNew = getWindow(particle,windowsize,frame)
            if (windowNew != -1):
                particlesValide.append(particle)
                rectangleNew = getRectangle(frame,windowNew)
                kmatNew = getKmat(rectangleNew,windowNew,b)
                fNew = 1/(np.sum(kmatNew))
                distributionNew = getDistributionMatrice(windowNew,rectangleNew,kmatNew,fNew,b)
                d = getDistance(distribution,distributionNew)
                weight = calculWeight(d,sigmaweight)
                weights.append(weight)

        weights = [float(i)/sum(weights) for i in weights]
        svs_predict = resampling(particlesValide,weights,sampleSize)
        svs_predict = addBruit(systemModel,sampleSize,dim,svs_predict)
        frame = draw_particles(svs_predict,frame)

        out.write(frame)
        #cv2.imshow('frame1',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        testcount-=1
    capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()