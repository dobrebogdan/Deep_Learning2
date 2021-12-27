# Submission accuracy: 0.52

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 128
img_width = 55
data_dir = './train'
num_classes = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(input_shape=(128, 55, 3), include_top=False, weights = 'imagenet')


for layer in base_model.layers:
    layer.trainable = False

# Flatten the output layer to 1 dimension
x = layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3))(base_model.output)

x = layers.Flatten()(x)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

x = layers.Dense(num_classes)(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

model.fit(train_ds, epochs = 50)

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
