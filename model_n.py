import csv
import numpy as np
import cv2
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.optimizers import Adam

lines = []
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

#augmented_images = []
#augmented_measurements = []
#for image, measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image, 1))
#    augmented_measurements.append(measurement * -1.0)

train_samples, validation_samples = train_test_split(lines[1:], test_size=0.1)


def generator(samples, batch_size=64):
    num_samples = len(samples)
    angle_correction = 0.2
    while True:
        shuffle(samples, random_state=42)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    images.append(cv2.flip(image, 1))
                    if i == 1:
                        angles.append(angle + angle_correction)
                        angles.append((angle + angle_correction) * -1.0)
                    elif i == 2:
                        angles.append(angle - angle_correction)
                        angles.append((angle - angle_correction) * -1.0)
                    else:
                        angles.append(angle)
                        angles.append(angle * -1.0)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train, random_state=42)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(1, 1, 1,  subsample=(1, 1), border_mode='same', init='glorot_uniform'))
model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 3, 3, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)
model.summary()
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*6,
                    nb_epoch=4,
                    verbose=1)
model.save('model.h5')

print('Done')
