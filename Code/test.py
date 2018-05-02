import numpy as np
import cv2
#import numpy as np;
import matplotlib.pylab as plt;
import scipy.ndimage
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from image_enhance import image_enhance


img = cv2.imread('Fake.png',0)
    
if(len(img.shape)>2):
    # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);

rows,cols = np.shape(img);
aspect_ratio = np.double(rows)/np.double(cols);

new_rows = 350;             # randomly selected number
new_cols = new_rows/aspect_ratio;

img = cv2.resize(img,(350,350));
#img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));

enhanced_img = image_enhance(img);
print enhanced_img.dtype
enhanced_img = 1 - enhanced_img
plt.imsave('filename.png', np.array(enhanced_img).reshape(350,350), cmap=cm.gray)
'''
print('saving the image')
cv2.imwrite('enhanced.png',enhanced_img)

    scipy.misc.imsave(img_name,enhanced_img);
else:
    plt.imshow(enhanced_img,cmap = 'Greys_r');
'''
