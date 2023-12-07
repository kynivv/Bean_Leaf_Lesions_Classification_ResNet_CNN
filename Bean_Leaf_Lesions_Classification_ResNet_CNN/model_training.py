import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from keras import layers
from glob import glob
from keras.applications import ResNet101
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 5
EPOCHS = 10
IMG_SIZE = 500
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Data Preprocessing
X_train = []
Y_train = []
X_test = []
Y_test = []

train_path = 'dataset/train'
test_path = 'dataset/val'

classes = os.listdir(train_path)

## Train Data
for i, name in enumerate(classes):
    images = glob(f'{train_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)
        X_train.append(img)
        Y_train.append(i)

## Test Data
for i, name in enumerate(classes):
    images = glob(f'{test_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)
        X_test.append(img)
        Y_test.append(i)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)


# Creating Model Based on ResNet
base_model = ResNet101(
    include_top= False,
    input_shape= IMG_SHAPE,
    pooling= 'max'
)

model = keras.Sequential([
    base_model,

    layers.Dense(3, activation= "softmax")
])

model.compile(
    optimizer= 'adam',
    loss= 'categorical_crossentropy',
    metrics= ['accuracy']
)


# Model Callbacks
checkpoint = ModelCheckpoint('output/finalmodel.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= False
                             )


# Model Training
history = model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          verbose= 1,
          validation_data= (X_test, Y_test),
          callbacks= checkpoint
          )


# Training Results Visualization
acc_train = history.history['acc']
acc_val = history.history['val_acc']
epochs = range(1,11)
plt.plot(epochs, acc_train, 'b', label='Training accuracy')
plt.plot(epochs, acc_val, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()