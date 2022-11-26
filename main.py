
from tensorflow import keras
from keras import layers as l
from keras import activations as af
import os

# used for the classification classes
numClasses = 3
inputShape = (227, 227, 3)

# model architecture
model = keras.Sequential(name="alexNet")
model.add(l.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=af.relu, input_shape=inputShape, padding='valid'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation=af.relu, input_shape=inputShape, padding='same'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape, padding='same'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape, padding='same'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape, padding='same'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.BatchNormalization())
model.add(l.Flatten())
model.add(l.Dense(units=4096, activation=af.relu))
model.add(l.Dropout(0.5))
model.add(l.Dense(units=4096, activation=af.relu))
model.add(l.Dropout(0.5))
model.add(l.Dense(units=1000, activation=af.relu))
model.add(l.Dense(units=numClasses, activation=af.softmax))
model.summary()

# compiling model
epochs = 50
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])

# load datasets
xTrain, yTrain = 0, 0
xTest, yTest = 0, 0
xValidate, yValidate = 0, 0

# training the model
# model.fit(epochs)