from PIL import Image
import glob
import numpy as np
import scipy as sp
import scipy.misc
import skimage.feature as ft
import pdb
import sys,os

i=1
print(i)

path = '/home/anusha/Desktop/wdt/'
dirlist = os.listdir(path)
print dirlist
'''
for folder in dirlist:
    print folder
    path1 = path + folder
    print path1
    dirlist = os.listdir(path1)
    print dirlist
'''
for file in dirlist:
    #print("Processing %s" % file)
    print (file)
    img = Image.open(os.path.join(path, file))
    im1 = sp.misc.imread(os.path.join(path, file))
    im = im1[:,:,0]
    print im
    ext='.png'
    name = str(i) + ext
    im = 255-im
    std = int(np.std(im)//1)
    rows = im.shape[0]
    cols = im.shape[1]
    #pdb.set_trace()
    x_axis = np.linspace(1,cols,num=cols)
    y_axis = np.linspace(1,rows,num=rows)
    x_com = np.tile(x_axis,(rows,1))
    y_com = np.tile(y_axis,(cols,1)).T
    mass = np.sum(im)
    x_coord = int(np.sum(x_com*im)/mass)
    y_coord = int(np.sum(y_com*im)/mass)
    left = int(max(0,x_coord-3*std))
    right = int(min(cols,x_coord+3*std))
    top =  max(0,y_coord-3*std)
    bottom = min(rows,y_coord+3*std)
    im = 255 - im
    image = im[top:bottom,left:right]
    output_file_name = os.path.join(path,"Extract" +file)
    scipy.misc.toimage(image,cmin=0.0,cmax= 255).save(output_file_name)
    i=i+1
    print(i)

