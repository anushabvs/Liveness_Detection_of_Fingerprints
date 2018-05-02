import cv2
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import scipy.ndimage
import sys
import pywt
from pywt import dwt2
import pandas as pd
from PIL import Image #Image Directory
import glob
import numpy as np
import scipy as sp
import scipy.misc
import skimage.feature as ft
import pdb
import sys,os
from scipy import signal
from Gabor import build_filters,process
from scipy import fftpack
from scipy import stats
from scipy.stats import norm
from image_enhance import image_enhance
from skimage import util
from skimage.morphology import skeletonize
from image_enhance import image_enhance

output_list = [] #initializing an array
wavelet_dec = []

#Input Image
if(len(sys.argv)<2):
    print('loading sample image');
    img_name = 'Extracted150.png'       
    img = scipy.ndimage.imread(img_name);
elif(len(sys.argv) >= 2):
    img_name = sys.argv[1];
    img = scipy.ndimage.imread(sys.argv[1]);
    
if(len(img.shape)>2):
    # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);

rows,cols = np.shape(img);
aspect_ratio = np.double(rows)/np.double(cols);
new_rows = 350;             # randomly selected number
new_cols = new_rows/aspect_ratio;
img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));
filters = build_filters() # Applying Gabor Filter
p = process(img, filters)
imgplot = plt.imshow(p)
plt.show()
enhanced_img = image_enhance(img); 
enhanced_img = skeletonize(enhanced_img)
#out_name = 'out5.png'   
#final_name = 'final5.png'
final_image = 1-enhanced_img
test = img * final_image
coeff = pywt.dwt2(test.T, 'db4')
cA, (cH, cV, cD) = coeff
for a in coeff:
	z = np.linalg.norm(a) 
	output_list.append(z) # Feature vector 1
Energy = (cH**2 + cV**2 + cD**2).sum()/test.size
output_list.append(Energy) # Feature vector 2
print Energy
name = 'test105.png'
print('saving the image')
scipy.misc.imsave(name,test)
test = test.astype(np.uint8)
f, t, Zxx = signal.stft(test) #Applying STFT
freq = f, t, np.real(Zxx)
print freq
test = test.ravel()
test = test.reshape((len(test),1))
output_list.append(np.std(test))# Feature vector 5
output_list.append(np.mean(test)) # Feature vector 6
t = test/270.0
plt.xlim(0,350)
plt.plot(t,'b')
plt.show()
'''
hist = cv2.calcHist([test],[0],None,[256],[0,350]) #calculating Histogram
hist = cv2.normalize(hist, None)
plt.plot(hist)
plt.show()
'''
w = pywt.Wavelet('dB10')
#fig, axes = plt.subplots(7, 1)
#fig.subplots_adjust(hspace = 1, wspace= None)
print pywt.dwt_max_level(len(test), w)
'''
axes[0].set_title('Original Signal')
axes[0].set_xlim([0, 350])
axes[0].plot(t,'b')
'''
coeffs = pywt.wavedec([test.ravel()], 'dB10', level=5)
[cA2,cD5,cD4,cD3,cD2,cD1] = coeffs

for i in coeffs:
        output_list.append(np.std(i))
        output_list.append(np.mean(i))
        print i.shape
        i = i.T
        print i.shape
        i = cv2.normalize(i,None)
        plt.xlim(0,350)
        #wavelet_dec.append(i)
        plt.plot(i,'r')
        plt.show()        
print len(output_list)
print output_list
#axes[7].set_title('feature_vector')
#axes[7].plot(output_list,'b')
plt.plot(output_list)
plt.show()
