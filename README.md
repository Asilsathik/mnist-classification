# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.


![image](https://github.com/Praveen0500/mnist-classification/assets/120218611/20cf93f6-5cd0-47bf-9a2e-2ecbfa5bed51)


## Neural Network Model

![Screenshot 2024-03-15 215150](https://github.com/Asilsathik/mnist-classification/assets/119476247/f358f4f2-190a-4e7e-99c1-4d8d8dfedb49)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:

Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM

### Name: MOHAMED ASIL M
### Register Number: 212222230080

```PY
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(7,7), input_shape=(28, 28, 1), activation='relu')),
model.add(layers.MaxPool2D(pool_size=(2, 2))),
model.add(layers.Conv2D(filters=20, kernel_size=(3,3), activation='relu')),
model.add(layers.MaxPool2D(pool_size=(2, 2))),
model.add(layers.Flatten()),
model.add(layers.Dense(64, activation='relu')),
model.add(layers.Dense(10, activation='softmax')) 

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/ex03 image.jpeg')

type(img)

img = image.load_img('/content/ex03 image.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)


```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-03-15 220201](https://github.com/Asilsathik/mnist-classification/assets/119476247/73b9b485-326a-4848-a7bd-87706a501604)


![Screenshot 2024-03-15 220214](https://github.com/Asilsathik/mnist-classification/assets/119476247/4149ac29-6c5f-4626-8ed0-94fff458888a)


### Classification Report

![Screenshot 2024-03-15 220424](https://github.com/Asilsathik/mnist-classification/assets/119476247/fec42a87-97f2-41d8-b94e-7e33b4fe5bc0)


### Confusion Matrix

![Screenshot 2024-03-15 220451](https://github.com/Asilsathik/mnist-classification/assets/119476247/0a3aadd9-83c3-4ded-be6d-42f275978030)

### New Sample Data Prediction

![Screenshot 2024-03-15 220610](https://github.com/Asilsathik/mnist-classification/assets/119476247/aedf07de-e99d-4fef-91cf-9e6acd5d6cb0)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
