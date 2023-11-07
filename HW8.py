import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from keras import layers
from keras import models
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import tensorflow
print(tensorflow.__version__)

train_dir = 'C:\\TMEIC\\VA\\Course_MLZoomCamp\\data\\data\\train'
test_dir = 'C:\\TMEIC\\VA\\Course_MLZoomCamp\\data\\data\\test'

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

###############Question 1
#Since we have a binary classification problem, what is the best loss function for us?
#Answer: binary crossentropy

model.compile(loss='binary_crossentropy',\
             optimizer=optimizers.SGD(learning_rate=0.002, momentum=0.8),\
             metrics=['acc'])

print(model.summary())

###############Question 2
#What's the number of parameters in the convolutional layer of our model? You can use the summary method for that.
#Answer: 11214912

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(test_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

###############Question 3
#What is the median of training accuracy for all the epochs for this model?
#Answer: 0.7968452572822571

acc_median = np.median(acc)
print(acc_median)

###############Question 4
#What is the standard deviation of training loss for all the epochs for this model?
#Answer: 0.09963254353568457

loss_std = np.std(loss)
print(loss_std)

datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150), 
                                                    batch_size=32, 
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)

acc_aug = history.history['acc']
val_acc_aug = history.history['val_acc']
loss_aug = history.history['loss']
val_loss_aug = history.history['val_loss']

epochs_aug = range(1, len(acc) + 1)

plt.plot(epochs_aug, acc_aug, 'bo', label='Training acc')
plt.plot(epochs_aug, val_acc_aug, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_aug, loss_aug, 'bo', label='Training loss')
plt.plot(epochs_aug, val_loss_aug, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

###############Question 5
#What is the mean of test loss for all the epochs for the model trained with augmentations?
#Answer: 0.4634510189294815

loss_mean_aug = np.mean(val_loss_aug)
print(loss_mean_aug)

###############Question 6
#What's the average of test accuracy for the last 5 epochs (from 6 to 10) for the model trained with augmentations?
#Answer: 0.7949890971183777

acc_mean_aug = np.mean(val_acc_aug[5:10])
print(acc_mean_aug)