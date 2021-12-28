# Submission accuracy: 0.629
# 25 epochs, accuracy: 0.62

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold

# variables for the program
batch_size = 32
img_height = 128
img_width = 55
depth = 3
data_dir = './train'
num_classes = 5
epochs = 25
n_splits = 5

""" 
Loading the dataset from the images directory. I split the directory in 5 subdirectories, one for each class.
Batch size is 15500 to get all the data in one go.
"""
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Getting the classnames
class_names = train_ds.class_names

# using the AUTOTUNE for the prefetch function to decide the number of prefetched elements at runtime
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
input_shape = (img_height, img_width, depth)
chanDim = -1

# Function used for getting the model
def get_model():
    # The sequential model and its layers
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
        layers.MaxPooling2D(pool_size=(2, 2), ),
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
        layers.MaxPooling2D(pool_size=(2, 2), ),
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
        layers.MaxPooling2D(pool_size=(2, 2), ),
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

    # Information about the model
    model.summary()
    return model

# Functions used for training the model and getting predictions
def train_and_predict():
    # getting the model
    model = get_model()
    # training the model
    model.fit(train_ds, epochs=epochs)
    # model.save('./model5')

    output_data = []
    # predicting and writing the data to the output file
    with open('test.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_path = f"test/{row[0]}"
            # reading an image
            img = tf.keras.utils.load_img(
                image_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch
            # prediction for the image
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            # Getting the label
            label = np.argmax(score)
            # Adding image id and label to the output
            output_data.append([row[0], int(class_names[label])])

    # opening the output file
    with open('output.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        # writing the output header
        writer.writerow(['id', 'label'])
        for output_row in output_data:
            # writing the id and
            writer.writerow(output_row)

# used for 5 fold cross validation
def n_fold_cross_validation():
    # Getting the data from the image directory.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=15500)
    train_data = np.array([])
    train_labels = np.array([])
    # Getting the train data and test data
    for x, y in train_ds:
        train_data = x.numpy()
        train_labels = y.numpy()
    # Using KFold to split in 5 parts for cross validation
    kfold = KFold(n_splits=n_splits, shuffle=True)
    kfold_split = kfold.split(train_data, train_labels)
    step = 0
    accuracies = []
    # iterating through the different splits
    for curr_train, curr_test in kfold_split:
        step += 1
        model = get_model()
        # training the model
        model.fit(train_data[curr_train], train_labels[curr_train], batch_size=batch_size, epochs=epochs)
        # getting the evaluation results
        results = model.evaluate(train_data[curr_test], train_labels[curr_test])
        # printing the loss and accuracy
        print(f'Step {step}: Loss - {results[0]}, Accuracy - {results[1]}')
        accuracies.append(results[1])
    # writing the results to an output file
    with open('results.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for accuracy in accuracies:
            writer.writerow(str(accuracy))

# Each of these functions serve a different purpose and should be commented or uncommented as needed
# train_and_predict()
n_fold_cross_validation()
