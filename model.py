import csv
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

images = []
measures = []

# load data
with open('./data/driving_log_all.csv') as csvfile:
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
        img_center = np.asarray(Image.open(path + row[0]))
        img_left = np.asarray(Image.open(path + row[1]))
        img_right = np.asarray(Image.open(path + row[2]))

        # add images and angles to data set
        images.extend([img_center, img_left, img_right])
        measures.extend([steering_center, steering_left, steering_right])

# flip image to generate more data
aug_images, aug_measures = [], []
for image, measure in zip(images, measures):
    aug_images.append(image)
    aug_measures.append(measure)
    aug_images.append(np.fliplr(image))
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
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# checkpoint
filepath="weights-{epoch:02d}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, nb_epoch=20, batch_size=128,
                    validation_split=0.2, shuffle=True,
                    callbacks=callbacks_list, verbose=1)

# save train loss and valid loss plot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('./loss_hist.png')
