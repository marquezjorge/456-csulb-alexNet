

import tensorflow as tf
from tensorflow import keras
from keras import layers as l
from keras import activations as af
import matplotlib.pyplot as plt
import datetime


# required variables
batchSize = 32
imgHeight = 227
imgWidth = 227
dataSet = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset"
trainDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\train"
testDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\dataset\\test"
validateDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\validate"

# --------------------

# model instance used for preprocessing images to be used in the datasets
dataAugmentation = keras.Sequential([l.RandomFlip("horizontal"), l.RandomRotation(0.2)])

# --------------------

tf.get_logger().setLevel('ERROR')
# loads images into 'train' and creates a semi-processed dataset
train = tf.keras.utils.image_dataset_from_directory(trainDir,
                                                 labels="inferred",
                                                 label_mode="int",
                                                 color_mode="rgb",
                                                 batch_size=batchSize,
                                                 image_size=(imgHeight, imgWidth))


plt.figure(figsize=(10, 10))
for imgs, _ in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

# standardizing 'train' data set, completes processing of the dataset using the dataAugmentation model
augmentedTrain = train.map(lambda x, y: (dataAugmentation(x, training=True), y))

# --------------------

# loads images into 'validate' and creates a semi-processed dataset
validate = tf.keras.utils.image_dataset_from_directory(validateDir,
                                                    labels="inferred",
                                                    label_mode="int",
                                                    color_mode="rgb",
                                                    batch_size=batchSize,
                                                    image_size=(imgHeight, imgWidth))
# standardizing 'validate' data set, completes processing of the dataset
augmentedValidation = validate.map(lambda x, y: (dataAugmentation(x, training=True), y))

# --------------------

test = tf.keras.utils.image_dataset_from_directory(testDir,
                                                    labels="inferred",
                                                    label_mode="int",
                                                    color_mode="rgb",
                                                    batch_size=batchSize,
                                                    image_size=(imgHeight, imgWidth))
augmentedTest = test.map(lambda x, y: (dataAugmentation(x, training=True), y))

# --------------------

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
model.add(l.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape,
                   padding='same'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=af.relu, input_shape=inputShape,
                   padding='same'))
model.add(l.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(l.Flatten())
model.add(l.Dense(units=4096, activation=af.relu))
model.add(l.Dense(units=4096, activation=af.relu))
model.add(l.Dense(units=1000, activation=af.relu))
model.add(l.Dense(units=numClasses, activation=af.softmax))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

# --------------------
# Creating a dir to store and visualize logs
logDir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorCallback = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=1)
fileWrite = tf.summary.create_file_writer(logDir + '/cm')
#callBack = keras.callbacks.LambdaCallback(on_epoch_end=logConfusionMatrix())


# --------------------
# training the model
epochs = 1
model.fit(augmentedTrain,
          epochs=epochs,
          validation_data=augmentedValidation,
          callbacks=[tensorCallback])

# --------------------

# metrics
metrics = model.evaluate(augmentedTest)
print("Test Accuracy:", metrics[1])


"""
rootLogDir = os.path.join(os.curdir, "logs\\fit\\")
def getLogRunDir():
    runID = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(rootLogDir, runID)

logRunDir = getLogRunDir()
tensorboardCB = keras.callbacks.TensorBoard(logRunDir)

# --------------------

callBack = cb.Callback()

"""