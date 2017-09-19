import os
import csv

import cv2
import numpy as np


samples = []

with open('/home/carnd/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0)#delete title

print('Current Set Size:',len(samples))


def data_merge(path, merge_set):
    tempo_set = []

    with open(path) as csvfile1:
        reader = csv.reader(csvfile1)
        for line in reader:
            tempo_set.append(line)
    print('subset', len(tempo_set))
    tempo_set.pop(0)  # delete title
    for sample in tempo_set:
        merge_set.append(sample)
    print('Current Set Size:', len(merge_set))


data_merge('/home/carnd/recovery/driving_log_recovery.csv', samples)
data_merge('/home/carnd/Curve/driving_log_curve.csv', samples)
data_merge('/home/carnd/lake/driving_log_lake.csv', samples)
data_merge('/home/carnd/Track2/driving_log_track2.csv', samples)
data_merge('/home/carnd/more/driving_log_more.csv', samples)


print('Total Set Size:', len(samples))

from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


def generator(samples, batch_size=75):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/home/carnd/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # image_flipped = np.fliplr(center_image)
                # measurement_flipped = -center_angle
                # images.append(image_flipped)
                # angles.append(measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print('generator is called correctly')
            # print(X_train.shape)
            # print(X_train[0])
            yield (X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=75)
validation_generator = generator(validation_samples, batch_size=75)

ch, row, col = 3, 160, 320  # Trimmed image format
stride = (2, 2)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# model.add(Dropout(0.2))

model.add(Conv2D(24, (5, 5), strides=stride, ))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(36, (5, 5), strides=stride))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(48, (5, 5), strides=stride))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(56, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

# model.load_weights('./mod17e.h5')
model.summary()

model.compile(loss='mse', optimizer='adam')
print('compile is ok')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator,
                    validation_steps=len(validation_samples), epochs=5)
print('Fit is OK')
model.save('model.h5')
model.summary()