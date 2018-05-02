import numpy as np;
import cv2;
from scipy import ndimage;
from scipy import signal

def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma):
    rows,cols = im.shape;
    #Calculate image gradients.
    sze = np.fix(6*gradientsigma);
    if np.remainder(sze,2) == 0:
        sze = sze+1;
        
    gauss = cv2.getGaussianKernel(np.int(sze),gradientsigma);
    f = gauss * gauss.T;
    
    fy,fx = np.gradient(f);     #Gradient of Gaussian
    
    #Gx = ndimage.convolve(np.double(im),fx);
    #Gy = ndimage.convolve(np.double(im),fy);
    
    Gx = signal.convolve2d(im,fx,mode='same');    
    Gy = signal.convolve2d(im,fy,mode='same');
    
    Gxx = np.power(Gx,2);
    Gyy = np.power(Gy,2);
    Gxy = Gx*Gy;
    
    #Now smooth the covariance data to perform a weighted summation of the data.    
    
    sze = np.fix(6*blocksigma);
    
    gauss = cv2.getGaussianKernel(np.int(sze),blocksigma);
    f = gauss * gauss.T;
    
    Gxx = ndimage.convolve(Gxx,f);
    Gyy = ndimage.convolve(Gyy,f);
    Gxy = 2*ndimage.convolve(Gxy,f);
    
    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps;
    
    sin2theta = Gxy/denom;            # Sine and cosine of doubled angles
    cos2theta = (Gxx-Gyy)/denom;
    
    
    if orientsmoothsigma:
        sze = np.fix(6*orientsmoothsigma);
        if np.remainder(sze,2) == 0:
            sze = sze+1;    
        gauss = cv2.getGaussianKernel(np.int(sze),orientsmoothsigma);
        f = gauss * gauss.T;
        cos2theta = ndimage.convolve(cos2theta,f); # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta,f); # doubled angles
    
    orientim = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2;
    return(orientim);
