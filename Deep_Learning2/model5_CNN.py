# Submission accuracy: 0.629

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 128
img_width = 55
depth = 3
data_dir = './train'
num_classes = 5
epochs = 25

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
input_shape = (img_height, img_width, depth)
chanDim = -1
model = Sequential([
    layers.Conv2D(64, (3, 3), padding="same", input_shape=input_shape),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(pool_size=(2, 2),),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), padding="same", input_shape=input_shape),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(pool_size=(2, 2),),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.ReLU(),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(pool_size=(2, 2),),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512),
    layers.ReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes),
    layers.Activation("softmax")
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

def train_and_predict():
    model.fit(train_ds, epochs=epochs)
    # model.save('./model5')

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

train_and_predict()
