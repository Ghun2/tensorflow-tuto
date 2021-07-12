import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

model = tf.keras.Sequential([
    # tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(28, 28, 1)),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='imageAndCnnModel/checkpoint/mnist',
    monitor='val_acc',
    mode='max',
    save_weights_only=True,
    save_freq='epoch'
)

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

from tensorflow.keras.callbacks import TensorBoard
import time

tensorboard = TensorBoard(log_dir='imageAndCnnModel/logs/{}'.format('firstModel' + str(int(time.time()))))

# model.load_weights('imageAndCnnModel/checkpoint/mnist')
#
# model.evaluate(testX, testY)

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[tensorboard])

# model.save('imageAndCnnModel/model1')

# score = model.evaluate(testX, testY)
# print(score)
