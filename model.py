import csv
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

images = []
measures = []

# load data
with open('./data/driving_log2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        row = []
        # load 3 dataset: center, right and left
        for i in range(3):
            row.append(line[i].split('/')[-1])
        path = './data/IMG/'

        steering_center = float(line[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = cv2.imread(path + row[0])
        img_left = cv2.imread(path + row[1])
        img_right = cv2.imread(path + row[2])

        # add images and angles to data set
        images.extend([img_center, img_left, img_right])
        measures.extend([steering_center, steering_left, steering_right])

# flip image to generate more data
aug_images, aug_measures = [], []
for image, measure in zip(images, measures):
    aug_images.append(image)
    aug_measures.append(measure)
    aug_images.append(cv2.flip(image, 1))
    aug_measures.append(measure*-1.0)

X_train = np.array(aug_images)
y_train = np.array(aug_measures)

print("X shape ", X_train.shape)
print("y shape ", y_train.shape)

model = Sequential()
# normalize image data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# crop off the top and bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# early stop to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model.fit(X_train, y_train,
          nb_epoch=4, batch_size=64,
          validation_split=0.2, shuffle=True,
          callbacks=[early_stop], verbose=1)
# save model
model.save('model.h5')
