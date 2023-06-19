# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 01:17:05 2023

@author: Rina
"""

#import all LIBRARIES

import glob #path
from skimage import io, filters, feature #scikit-image
import matplotlib.pyplot as plt #plot image 
from skimage.color import rgb2gray #rgb 
import cv2
import numpy as np #array purposes

#path = "images/test_images/*.*"
file_list = glob.glob('bird data training set (95 pic each)/Asian Openbill/*.*') #Rerurns a list of file names
print(file_list)  #Prints the list containing file names

# this one can make our image from each folder save in our my_list
#Now let us load each file at a time...
my_list=[]  #Empty list to store images from the folder.

path = "bird data training set (95 pic each)/Asian Openbill/*.*"
for file in glob.glob(path):   
    #Iterate through each file in the list using for
    print(file)     
    #just stop here to see all file names printed
    a= cv2.imread(file) 
    #now, we can read each file since we have the full path
    my_list.append(a) 
    #Create a list of images (not just file names but full images)
    
#View images from the stored list
plt.imshow(my_list[9])  
#View the 3rd image in the list.

#Edge detection
img = cv2.imread('C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Asian Openbill/Image_1.jpg', 0)
from skimage.filters import roberts

roberts_img = roberts(img)

cv2.imshow("Roberts", roberts_img)
cv2.waitKey(0)
cv2.destroyAllWindows()










#