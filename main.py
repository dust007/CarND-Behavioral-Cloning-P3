import csv
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import numpy as np

lines = []
with open('./data/driving_log2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measures = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    # print(current_path)
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measures.append(measurement)

X_train = np.array(images)
y_train = np.array(measures)

print("X shape ", X_train.shape)
print("y shape ", y_train.shape)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
