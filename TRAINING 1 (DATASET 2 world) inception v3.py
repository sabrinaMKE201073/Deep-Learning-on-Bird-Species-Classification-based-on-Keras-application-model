# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:11:41 2023

@author: Rina
"""

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
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


#add preprocessing layer to the front of Inception V3
# Here we will be using imagenet weights
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#don't train existing weights
for layer in inception.layers:
    layer.trainable = False
    
#to get number of output classes
folders = glob('dataset 2 (kaggle world chosen bird species)/train/*')
    
#our additional layers
x = Flatten()(inception.output)    
prediction = Dense(len(folders), activation='softmax')(x)

#create a model object
model = Model(inputs=inception.input, outputs=prediction)   

#compile the model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
    
#Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)    
    
    
#training and test sets
training_set = train_datagen.flow_from_directory('dataset 2 (kaggle world chosen bird species)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset 2 (kaggle world chosen bird species)/valid',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')



# fit the model and running time
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


#analysis of accuracy and loss for training and validation data
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

from tensorflow.keras.models import load_model

model.save('model_inception.h5')

y_pred = model.predict(test_set)

y_pred


import numpy as np
y_pred = np.argmax(y_pred, axis=1)

y_pred

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model=load_model('model_inception.h5')

img=image.load_img('dataset 2 (kaggle world chosen bird species)/valid/BALD EAGLE/1.jpg',target_size=(224,224))

x=image.img_to_array(img)

x

x.shape

x=x/255

import numpy as np
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


model.predict(img_data)

a=np.argmax(model.predict(img_data), axis=1)

a==1











