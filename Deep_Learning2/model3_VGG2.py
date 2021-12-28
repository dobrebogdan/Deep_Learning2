# Submission accuracy: 0.52

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import KFold

batch_size = 32
img_height = 128
img_width = 55
data_dir = './train'
num_classes = 5
epochs = 15
n_splits = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

def get_model():
    vgg_model = VGG16(input_shape=(128, 55, 3), include_top=False, weights='imagenet')
    for layer in vgg_model.layers:
        layer.trainable = False

    # Flatten the output layer to 1 dimension
    custom_layers = [
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
    ]

    output = vgg_model.output
    for layer in custom_layers:
        output = layer(output)

    model = tf.keras.models.Model(vgg_model.input, output)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def train_and_predict():
    model = get_model()
    model.fit(train_ds, epochs=epochs)
    # model.save('./model3')

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

def n_fold_cross_validation():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=15500)
    train_data = np.array([])
    train_labels = np.array([])
    for x, y in train_ds:
        train_data = x.numpy()
        train_labels = y.numpy()
    kfold = KFold(n_splits=n_splits, shuffle=True)
    kfold_split = kfold.split(train_data, train_labels)
    step = 0
    accuracies = []
    for curr_train, curr_test in kfold_split:
        step += 1
        model = get_model()
        model.fit(train_data[curr_train], train_labels[curr_train], batch_size=batch_size, epochs=epochs)
        results = model.evaluate(train_data[curr_test], train_labels[curr_test])
        print(f'Step {step}: Loss - {results[0]}, Accuracy - {results[1]}')
        accuracies.append(results[1])
    with open('results.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for accuracy in accuracies:
            writer.writerow(str(accuracy))


# train_and_predict()
n_fold_cross_validation()
