# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:45:18 2023

@author: Rina
"""

"""
INSTALL LIBRARIES AND DEPENDENCIES
"""

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


"""
SET IMAGE SIZE FOR OUR CLASSIFICATION MODEL = 224*224*3 FOR ALL IMAGE DATASETS
"""
IMAGE_SIZE = [224, 224]

train_path = '/content/drive/MyDrive/dataset 2 (kaggle world chosen bird species)/train'
valid_path = '/content/drive/MyDrive/dataset 2 (kaggle world chosen bird species)/valid'


"""
SETUP THE LAYERS FOR OUR DEEP LEARNING MODEL
"""
#add preprocessing layer to the front of Inception V3
# Here we will be using imagenet weights
DenseNet121 = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#don't train existing weights
for layer in DenseNet121.layers:
    layer.trainable = False
    
folders = glob('/content/drive/MyDrive/dataset 2 (kaggle world chosen bird species)/train/*')

#our additional layers
x = Flatten()(DenseNet121.output)    
prediction = Dense(len(folders), activation='softmax')(x)

#create a model object
model = Model(inputs=DenseNet121.input, outputs=prediction)   

#compile the model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


"""
DATA AUGMENTATION METHOD
"""
#Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.3,
                                   horizontal_flip = True,
                                   rotation_range=90)

test_datagen = ImageDataGenerator(rescale = 1./255)    
    
    
"""
PREPARING DATASET PATH FOR TRAINING AND VALIDATION: RESIZE, SET THE BATCH SIZE, CLASS MODE
"""
#training and test sets
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/dataset 2 (kaggle world chosen bird species)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle=False)

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/dataset 2 (kaggle world chosen bird species)/valid',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            shuffle=False)


#Get the labels for the training and validation sets
train_labels = training_set.classes
test_labels = test_set.classes

"""
TRAINING TIME
"""
# fit the model and running time
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


"""
PLOT THE PERFORMANCE OF OUR CLASSIFICATION MODEL
"""
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


"""
ACCURACY FOR EACH CLASS
"""
#calculate accuracy for each category
from sklearn.metrics import accuracy_score

train_acc = []
test_acc = []

for i in range(len(folders)):
    category = folders[i].split('/')[-1]
    train_category_acc = accuracy_score(train_labels, train_labels == i)
    test_category_acc = accuracy_score(test_labels, test_labels == i)
    train_acc.append(train_category_acc)
    test_acc.append(test_category_acc)
    print(f"Category: {category} \tTrain accuracy: {train_category_acc:.3f} \tTest accuracy: {test_category_acc:.3f}")
    
    


"""
PLOT CONFUSION MATRIX FOR EACH CLASSES
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate predictions for the test set
predictions = model.predict_generator(test_set)

# Get the predicted class for each image
predicted_classes = np.argmax(predictions, axis=1)

# Get the actual class for each image
true_classes = test_set.classes

# Compute the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()



"""
SAVE THE MODEL
"""
from tensorflow.keras.models import load_model

model.save('correctmodel_densenetdata2.h5')











