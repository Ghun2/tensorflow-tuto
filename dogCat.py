import tensorflow as tf
import os
import shutil

trainDir = './dogCat/dataset/train/'

datasetDir = './dogCat/dataset/'

catDir = './dogCat/dataset/cat/'
dogDir = './dogCat/dataset/dog/'

# for i in os.listdir(trainDir):
#     if 'cat' in i:
#         shutil.copyfile(trainDir+i, catDir+i)
#     if 'dog' in i:
#         shutil.copyfile(trainDir+i, dogDir+i)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    datasetDir,
    image_size=(64, 64),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=1234,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    datasetDir,
    image_size=(64, 64),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=1234,
)


def preprocess_function(i, answer):
    i = tf.cast(i/255.0, tf.float32)
    return i, answer


train_ds = train_ds.map(preprocess_function)
val_ds = val_ds.map(preprocess_function)

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64, 64, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save('dogCat/dogCatModel/model1')
