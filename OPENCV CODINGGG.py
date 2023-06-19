# -*- coding: utf-8 -*-
"""
~~~ USING OPEN CV (CODING LIST) ~~~~~~

1) READ IMAGES (TIF,JPG,PNG)
"""

import cv2
img = cv2.imread('C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Eurasian Wigeon duck/Image_11.jpg')
 

"""
2) RESIZE IMAGE
"""
import cv2
 
img = cv2.imread('C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Eurasian Wigeon duck/Image_11.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0) #UNTIL I CLOSED IT, LEAVE IT OPEN
cv2.destroyAllWindows()

"""
3) CHANGE COLOR
"""

import cv2

gray_img = cv2.imread("C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Eurasian Wigeon duck/Image_11.jpg", 0)
color_img = cv2.imread("C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Eurasian Wigeon duck/Image_11.jpg", 1)

cv2.imshow("color pic from opencv", color_img)
cv2.imshow("gray pic from opencv", gray_img)

# Maintain output window until 
# user presses a key or 1000 ms (1s)
cv2.waitKey(0)          

#destroys all windows created
cv2.destroyAllWindows() 

"""
4) EDGE DETECTION
"""

import cv2
 
img = cv2.imread('C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Eurasian Wigeon duck/Image_11.jpg')
edges = cv2.Canny(img,100,200)
 
cv2.imshow("Edge Detected Image", edges)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

"""
4) GAUSSIAN BLUR (IMAGE SMOOTHING)
"""

import cv2
import numpy
  
# read image
src = cv2.imread('C:/Users/nurul/.spyder-py3/bird data training set (95 pic each)/Eurasian Wigeon duck/Image_11.jpg', cv2.IMREAD_UNCHANGED)
 
# apply guassian blur on src image
dst = cv2.GaussianBlur(src,(5,5),cv2.BORDER_DEFAULT)
 
# display input and output image
cv2.imshow("Gaussian Smoothing",numpy.hstack((src, dst)))
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

"""
4) GAUSSIAN BLUR
"""











