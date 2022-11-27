from tensorflow import keras
from keras import layers as l
from keras import activations as af
import os
import shutil

# used for the classification classes
numClasses = 3
inputShape = (227, 227, 3)

# model architecture
model = keras.Sequential(name="alexNet")
model.add(l.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=af.relu, input_shape=inputShape,
                   padding='valid'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation=af.relu, input_shape=inputShape,
                   padding='same'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape,
                   padding='same'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape,
                   padding='same'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape,
                   padding='same'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.BatchNormalization())
model.add(l.Flatten())
model.add(l.Dense(units=4096, activation=af.relu))
model.add(l.Dropout(0.5))
model.add(l.Dense(units=4096, activation=af.relu))
model.add(l.Dropout(0.5))
model.add(l.Dense(units=1000, activation=af.relu))
model.add(l.Dense(units=numClasses, activation=af.softmax))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])

# required variables
epochs = 50
batchSize = 32
imgHeight = 227
imgWidth = 227
dataSet = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset"
trainDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\train"
testDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\dataset\\test"
validateDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\validate"

train = keras.utils.image_dataset_from_directory(trainDir,
                                                 labels="inferred",
                                                 label_mode="int",
                                                 color_mode="rgb",
                                                 batch_size=batchSize,
                                                 image_size=(imgHeight, imgWidth))

validate = keras.utils.image_dataset_from_directory(trainDir,
                                                    labels="inferred",
                                                    label_mode="int",
                                                    color_mode="rgb",
                                                    batch_size=batchSize,
                                                    image_size=(imgHeight, imgWidth))

# training the model
model.fit(train,
          epochs=epochs,
          validation_data=validate)

