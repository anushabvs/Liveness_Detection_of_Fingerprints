# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""
import cv2
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import scipy.ndimage
import sys
import pywt
#import pandas as pd
from PIL import Image #Image Directory
import csv
import glob
import numpy as np
import scipy as sp
import scipy.misc
import skimage.feature as ft
import pdb
import sys,os
from Gabor import build_filters,process
from sklearn import preprocessing
from scipy import fftpack
from scipy import stats
from scipy import signal
from scipy.stats import norm
from image_enhance import image_enhance
from skimage import util
from skimage.morphology import skeletonize
from image_enhance import image_enhance

i=1
print(i)
path = '/home/anusha/Desktop/Wavelet_data/Training/Live/'
dirlist = os.listdir(path)

header = ["Image","L2CA","L2CD","Energy","Original std", "Original mean", "cA5 std", "cA5 mean","cD5 std", "cD5 mean",
                          "cD4 mean","cD4 std","cD3 mean","cD3 std",
                         "cD2 mean","cD2 std", "cD1 mean","cD1 std","Class"]
with open("Trainngcoif.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(header)

for file in dirlist:
    #print("Processing %s" % file)
    output_list = [] #initializing an array
    wavelet_dec = []
    print (file)
    output_list.append(file)#1
    img = Image.open(os.path.join(path, file))
    image = sp.misc.imread(os.path.join(path, file)) 
    #Input Image
    '''
    if(len(sys.argv)<2):
        print('loading sample image');
        img_name = image       
        #img = scipy.ndimage.imread(img_name);
    elif(len(sys.argv) >= 2):
        img_name = sys.argv[1];
        #img = scipy.ndimage.imread(sys.argv[1]);
        
    if(len(img.shape)>2):
        # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);
    '''
    image = img
    rows,cols = np.shape(img);
    aspect_ratio = np.double(rows)/np.double(cols);
    new_rows = 350;             # randomly selected number
    new_cols = new_rows/aspect_ratio;
    img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));
    filters = build_filters() # Applying Gabor Filter
    p = process(img, filters)
    enhanced_img = image_enhance(img); 
    enhanced_img = skeletonize(enhanced_img)
    out_name = 'out1.png'   
    final_name = 'final1.png'
    final_image = 1-enhanced_img
    test = img * final_image
    name = 'test110.png'
    print('saving the image')
    scipy.misc.imsave(name,test)
    coeff = pywt.dwt2(test.T, 'db4')
    cA, (cH, cV, cD) = coeff
    for a in coeff:
    	z = np.linalg.norm(a) 
    	output_list.append(z) # Feature vector 1
    Energy = (cH**2 + cV**2 + cD**2).sum()/test.size
    output_list.append(Energy) # Feature vector 2
    print Energy
    test = test.astype(np.uint8)
    test = test.ravel()
    test = test.reshape((len(test),1))
    output_list.append(np.std(test))
    output_list.append(np.mean(test))
   
    w = pywt.Wavelet('coif10')
    
    print pywt.dwt_max_level(len(test), w)
    coeffs = pywt.wavedec([test.ravel()], 'coif10', level=5)
    [cA5,cD5,cD4,cD3,cD2,cD1] = coeffs

    for i in coeffs:
            i.shape
            output_list.append(np.std(i))
            output_list.append(np.mean(i))
    output_list.append('Live')       
    print len(output_list)
    print output_list
    
    with open("Trainngcoif.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(output_list)
    '''
    if(1):
        print('saving the image')
     
        scipy.misc.imsave(out_name,enhanced_img);
        scipy.misc.imsave(final_name,final_image);
        
    else:
        plt.imshow(enhanced_img,cmap = 'Greys_r');

    '''





