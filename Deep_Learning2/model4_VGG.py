# Submission accuracy: 0.52

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = height = 128
img_width = width =  55
depth = 3
data_dir = './train'
num_classes = classes = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1

# CONV => RELU => POOL layer set
model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
# (CONV => RELU) * 2 => POOL layer set
model.add(layers.Conv2D(64, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.Conv2D(64, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
# (CONV => RELU) * 3 => POOL layer set
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
# (CONV => RELU) * 3 => POOL layer set
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization(axis=chanDim))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
# first (and only) set of FC => RELU layers
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation("relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
#softmax classifier
model.add(layers.Dense(classes))
model.add(layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 20
history = model.fit(
    train_ds,
    epochs=epochs
)
model.save('./model')

output_data = []
with open('test.csv') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        image_path = f"test/{row[0]}"
        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        label = np.argmax(score)
        output_data.append([row[0], int(class_names[label])])

with open('output.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['id', 'label'])
    for output_row in output_data:
        writer.writerow(output_row)