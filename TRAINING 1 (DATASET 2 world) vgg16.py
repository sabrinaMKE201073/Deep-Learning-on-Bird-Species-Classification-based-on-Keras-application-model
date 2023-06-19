# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:06:13 2023

@author: Rina
"""

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'dataset 2 (kaggle world chosen bird species)/train'
valid_path = 'dataset 2 (kaggle world chosen bird species)/valid'


# Import the Inception V3 library as shown below and add preprocessing layer to the front of Inception V3
# Here we will be using imagenet weights

Vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in Vgg16.layers:
    layer.trainable = False
    
    
# useful for getting number of output classes
folders = glob('dataset 2 (kaggle world chosen bird species)/train/*')
    
# our layers - you can add more if you want
x = Flatten()(Vgg16.output)    

 
prediction = Dense(len(folders), activation='softmax')(x)



# create a model object
model = Model(inputs=Vgg16.input, outputs=prediction)   


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
    
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)    
    
    
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('dataset 2 (kaggle world chosen bird species)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset 2 (kaggle world chosen bird species)/valid',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')



# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


import matplotlib.pyplot as plt

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')