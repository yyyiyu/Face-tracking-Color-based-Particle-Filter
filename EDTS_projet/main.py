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
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class SystemModel:
    def __init__(self,model):
        self.model = model
    def generate(self,sv,w):
        return(self.model(sv,w))

def model_s(sv,w):
    """Propagate the sample, a state vector, through a dynamic model """

    # First order model describing an object moving wiht constant velocity for x,y,hx,hy
    A = np.matrix([[1,0,1,0,0,0,0],
                   [0,1,0,1,0,0,0],
                   [0,0,1,0,0,0,0],
                   [0,0,0,1,0,0,0],
                   [0,0,0,0,1,0,1],
                   [0,0,0,0,0,1,1],
                   [0,0,0,0,0,0,1]])
    
    # s(t) = A*s(t-a)+w(t-1)
    return(np.array(np.dot(A,sv))[0]+w)


def getWindow(pt,windowsize,frame):
    """Calculate new window with current locate x, y"""

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

def getYuvWindow(frame,window):
    """Generate luminance rectangle for tracked object """

    yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV);
    rectangle = yuv[window[0]:window[2],window[1]:window[3]]

    return rectangle

def calculateColorDistribution(rectangle,yuvChannel,colorBin,kmat,f):
    pY = 0

    for i in range(rectangle.shape[0]):
        for j in range(rectangle.shape[1]):
            tmp = rectangle[i][j][yuvChannel]
            if (np.ceil(tmp/32)==colorBin):
                pY = pY + kmat[i][j]*f
    return pY

def getDistributionMatrice(rectangle,kmat,f):
    # 3 channels, 8 bins for color distribution
    distribution = np.zeros((3,8)) 
    for i in range(3):
        for j in range(8):
            distribution[i][j]=calculateColorDistribution(rectangle,i,j+1,kmat,f)
    return distribution


def getPixelPosition(rectangle,windows):
    rectanglepts = np.ones((rectangle.shape[0],rectangle.shape[1],1))*(1,1)
    for i in range(0,rectangle.shape[0],1):
        for j in range(0,rectangle.shape[1],1):
            rectanglepts[i,j] = (i+windows[0],j+windows[1])
    return rectanglepts


def getKmat(rectangle,window,b):
    """Calculation of the weighting for each particle"""
    rectanglepts = getPixelPosition(rectangle,window)
    # center of rectangle
    c = [(window[2]+window[0])/2,(window[3]+window[1])/2] # y, x
    # color distribution
    cmat = np.ones((rectanglepts.shape[0],rectanglepts.shape[1],1))*c
    rmat = np.divide(distancesPts(rectanglepts,cmat),b)
    # weighting functino
    kmat = np.ones((rectangle.shape[0],rectangle.shape[1])) - rmat**2
    return kmat

def initStateVectors(window,sampleSize):
    xs = [random.uniform(window[1],window[3]) for i in range(sampleSize)]
    ys = [random.uniform(window[0],window[2]) for i in range(sampleSize)]
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
    """ Calculate Bhattacharyya distance between two distributions"""
    # calculate distance for each channel
    d = np.zeros((1,3))
    for i in range(3):
        # when bcoef = 1 two distributions are identical
        bcoef =sum(np.sqrt(distribution[i]*distributionNew[i]))
        # the distance bhattacharyya defined in reference paper, to understand why there is a difference  
        d[0][i] = np.sqrt(1-bcoef)
    return d

def calculWeight(d,sigma):
    dprime = d.transpose()
    sigmadet = np.linalg.det(sigma)
    sigmainv = np.linalg.inv(sigma)
    weight = np.exp((-1/2)*(np.dot((np.dot(d,sigmainv)),dprime))) / (np.sqrt(np.power(2*np.pi,3)*sigmadet))
    return weight[0]

def resampling(svs,weights,N):
    sorted_particle = sorted([list(x) for x in zip(svs,weights)],key=lambda x:x[1],reverse=True)
    logging.info(f"Nb sorted particles : {len(sorted_particle)}")
    logging.info(f"first 10 particles' weight : {sorted(weights[:10],reverse=True)}")
    resampled_particle = []
    while(len(resampled_particle)<N):
        for sp in sorted_particle:
            resampled_particle.append(sp[0])
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

def initialisation(image,window,sampleSize=30):
    # window diagonal distance
    b = np.sqrt(np.power((window[2]-window[0]),2) + np.power((window[3]-window[1]),2))
    # sigma filter
    sigma = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # generate state vector for samples
    svs = initStateVectors(window,sampleSize)
    # get yuv version for target window
    rectangle = getYuvWindow(image,window)
    # initialize weighting
    kmat = getKmat(rectangle,window,b)
    # normalisation factor
    f = 1/(np.sum(kmat))
    distribution = getDistributionMatrice(rectangle,kmat,f)

    return (b,sigma,svs,distribution)

def main():
    # Define output video file
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter('output/yL1uOzsjXlo_000003_000013_ft.avi',fourcc, 20, (480,360))

    # Load input video
    capture = cv2.VideoCapture('video/yL1uOzsjXlo_000003_000013.avi')
    # Initial rectangle
    imageSize=[capture.get(cv2.CAP_PROP_FRAME_HEIGHT),capture.get(cv2.CAP_PROP_FRAME_WIDTH)]
    logging.info(f"Input video frame size H#{imageSize[0]}, W#{imageSize[1]}")
    ret, image = capture.read()
    logging.info(f"capture read image succeed: {ret}")


    # Initial the window that contains the object to be traced, for the input video, we track the face of a climber
    window = [90, 220, 180, 280] # y1, x1, y2, x2
    windowsize = [window[2]-window[0],window[3]-window[1]] #Hy, Hx
    logging.info(f"Initialized object window {window}, size H#{windowsize[0]}, W#{windowsize[1]}")

    # Initialize particle samples and their distribution
    sampleSize=30
    (b,sigmaweight,svs,distribution) = initialisation(image,window,sampleSize)


    # Uncomment to show the initialized object to be tracked
    # # Draw particles on frame and show frame
    # frameInit = draw_particles(svs,image)
    # cv2.imshow('frame0',frameInit)
    # cv2.imshow('frame1',frameInit[window[0]:window[2],window[1]:window[3]])
    # cv2.waitKey() 

    dim = len(svs[1])
    while(capture.isOpened()):
        ret, frame = capture.read()
        systemModel = SystemModel(model_s)
        svs_predict = svs 
        weights = []
        particlesValide = []
        for particle in svs_predict:
            windowNew = getWindow(particle,windowsize,frame)
            if (windowNew != -1):
                particlesValide.append(particle)
                rectangleNew = getYuvWindow(frame,windowNew)
                kmatNew = getKmat(rectangleNew,windowNew,b)
                fNew = 1/(np.sum(kmatNew))
                distributionNew = getDistributionMatrice(rectangleNew,kmatNew,fNew)
                d = getDistance(distribution,distributionNew)
                weight = calculWeight(d,sigmaweight)
                weights.append(weight)

        weights = [float(i)/sum(weights) for i in weights]
        svs_predict = resampling(particlesValide,weights,sampleSize)
        svs_predict = addBruit(systemModel,sampleSize,dim,svs_predict)
        frame = draw_particles(svs_predict,frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()