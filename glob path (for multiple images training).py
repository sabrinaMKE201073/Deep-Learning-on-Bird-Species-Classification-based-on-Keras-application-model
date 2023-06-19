# -*- coding: utf-8 -*-
"""
using GLOB to read multiple files in Python
"""

"""

### Reading multiple images from a folder
#The glob module finds all the path names 
#matching a specified pattern according to the rules used by the Unix shell
#The glob.glob returns the list of files with their full path 

"""


"""
INPUT MULTPLE IMAGE (LIST IN ARRAYS)
"""

#import the library opencv
import cv2
import glob

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
from matplotlib import pyplot as plt
plt.imshow(my_list[9])  
#View the 3rd image in the list.



"""
AFTER IMAGE IS PROCESS, WE WANT TO PUT THE IMAGE INTO RELATED FOLDER/PATH
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

#select the path
path = "bird data training set (95 pic each)/Daurian Redstart/*.*"
img_number = 1  #Start an iterator for image number.
#This number can be later added to output image file names.

#using glob.glob(path) to extract filename 
for file in glob.glob(path):
    #print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    #print(a)  #print numpy arrays for each file

#let us look at each file
#    cv2.imshow('Original Image', a)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#process each image - change color from BGR to RGB.

    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  
    #Change color space from BGR to RGB
    cv2.imwrite("bird data training set (95 pic each)/Daurian Redstart/Color_image"+str(img_number)+".jpg", c)
    img_number +=1 
    cv2.imshow('Color image', c)
    cv2.waitKey(10)  #Display each image for 1 second = 1000
    cv2.destroyAllWindows()
    


#have try below but not works
"""  
#process each image (feature gabor filters)
ksize = 8  #Use size that makes sense to the image and fetaure size. Large may not be good. 
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5 #Large sigma on small features will fully miss the features. 
theta = 1*np.pi/4  #/4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1*np.pi/4  #1/4 works best for angled. 
gamma=0.9 #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
#Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0.8  #Phase offset. I leave it to 0. (For hidden pic use 0.8)


kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
cv2.imwrite("bird data training set (95 pic each)/Daurian Redstart/kernel_image"+str(img_number)+".jpg", kernel)
img_number +=1 
#cv2.imshow('Color image', kernel)
#cv2.waitKey(10)  #Display each image for 1 second = 1000
#cv2.destroyAllWindows()
"""






























