import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as plt
import sklearn
import random
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.visualize_util import plot


samples=[]
y_steer=[]
nepochs = 5
#Store the rows from csv file
with open('data/driving_log.csv', 'rt') as drivelog:
	reader = csv.reader(drivelog)
	for i,row in enumerate(reader):
		if i > 0:
			samples.append(row)
			y_steer.append(float(row[3]))

print("Data set count",i)

#Plot histogram of steeting angles
n,bins,patch=plt.hist(y_steer, 30, histtype='bar')
print(n)
print(bins)
plt.title("Histogram with steering angles")
plt.show()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def random_brightness(image):
    #Convert from RGB to HSV color space
    image_temp = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    #randomly generate a brightness multiplication factor
    brightness = 0.25 + np.random.uniform()

    #apply the brightness factor to the brightness plane
    image_temp[:,:,2]=image_temp[:,:,2]*brightness

    #clip the pixel values greater than 255
    image_temp = np.clip(image_temp,0,255)

    #convert back from HSV to RGB
    image_temp = cv2.cvtColor(image_temp,cv2.COLOR_HSV2RGB)
    return image_temp

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.25
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_index = random.randint(0,2)
                # 0-center image 1-left image 2-right image
                image = cv2.imread('data/'+batch_sample[image_index].lstrip())
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                
                angle = float(batch_sample[3])
                
                #if left image shift the steering angle to right
                #if right image shift the steering anfle to left
                if (image_index == 1): 
                    angle = angle + correction
                elif (image_index == 2):
                    angle = angle - correction

                image = random_brightness(image)
                flip_prob = random.randint(0,1)
                #if flip_prob = 1, flip the image and apply the negative steering angle
                if (flip_prob):
                    image = cv2.flip(image,1)
                    angle = (-1)*angle

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

inp_shape=(160,320,3)
#Nvidia Network Architecture
model=Sequential()
model.add(Lambda(lambda x: ((x/255.0) - 0.5), input_shape=inp_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

#Save the model architecture image
plot(model, to_file='model.png')

#compile the model for adam optimizer and minimize mse
model.compile(optimizer='adam', loss='mean_squared_error')

#use model generator for on the fly augmentation and training
#store the history of the model for plotting the loss function accross epochs
history_object=model.fit_generator(train_generator, samples_per_epoch= 67000, validation_data=validation_generator, nb_val_samples=16751, nb_epoch=nepochs)

#save the model
model.save('model.h5')

#plot the loss function w.r.t epochs
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
