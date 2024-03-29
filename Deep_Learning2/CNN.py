# Importing libraries
import csv
from keras.optimizer_v2.adam import Adam
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# some constants for the program
batch_size = 32
imag_height = 128
imag_width = 55
depth = 3
training_directory = './train'
classes_number = 5
epochs_number = 25
splits_number = 5

""" 
Loading the dataset from the images directory. I split the directory in 5 subdirectories, one for each class.
Batch size is 15500 to get all the data in one go.
"""
train_dataset = tf.keras.utils.image_dataset_from_directory(
    training_directory,
    seed=17,
    image_size=(imag_height, imag_width),
    batch_size=batch_size)

# Getting the classnames
class_names = train_dataset.class_names

# using the AUTOTUNE for the prefetch function to decide the number of prefetched elements at runtime
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
input_shape = (imag_height, imag_width, depth)
normalization_axis = -1

# Function used for getting the model
def get_model(learning_rate=None):
    # The sequential model and its layers
    model = Sequential([
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(64, (3, 3), padding="same", input_shape=input_shape),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Max pooling and dropout layers
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(128, (3, 3), padding="same", input_shape=input_shape),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Max pooling and dropout layers
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Conv2D, Relu and Batch normalization layers
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(axis=normalization_axis),
        # Max pooling and dropout layers
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        # Flatten layer
        layers.Flatten(),
        # Dense layer
        layers.Dense(512),
        # Relu and Batch normalization layers
        layers.ReLU(),
        layers.BatchNormalization(),
        # Dropout layer
        layers.Dropout(0.5),
        # Dense layer with 5 nodes
        layers.Dense(classes_number),
        # Softmax activation layer
        layers.Activation("softmax")
    ])
    optimizer = 'adam'
    # Setting Adam optimizer with the right learning rate if we get one
    if learning_rate:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Information about the model
    model.summary()
    return model

# Functions used for training the model and getting predictions
def train_and_predict():
    # getting the model
    model = get_model()
    # training the model
    model.fit(train_dataset, epochs=epochs_number)
    # model.save('./cnn_model')

    output_data = []
    # predicting and writing the data to the output file
    with open('test.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_path = f"test/{row[0]}"
            # reading an image
            img = tf.keras.utils.load_img(image_path, target_size=(imag_height, imag_width))
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
    train_dataset = tf.keras.utils.image_dataset_from_directory(training_directory, image_size=(imag_height, imag_width),
                                                                batch_size=15500)
    train_data = np.array([])
    train_labels = np.array([])
    # Getting the train data and test data
    for x, y in train_dataset:
        train_data = x.numpy()
        train_labels = y.numpy()
    # Using KFold to split in 5 parts for cross validation
    kfold = KFold(n_splits=splits_number, shuffle=True)
    kfold_split = kfold.split(train_data, train_labels)
    step = 0
    accuracies = []
    # iterating through the different splits
    for curr_train, curr_test in kfold_split:
        step += 1
        model = get_model()
        # training the model
        model.fit(train_data[curr_train], train_labels[curr_train], batch_size=batch_size, epochs=epochs_number)
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

# function used for searching for an optimal learning rate
def grid_search_learning_rate():
    """
        Loading the dataset from the images' directory. I split the directory in 5 subdirectories, one for each class.
        Batch size is 15500 to get all the data in one go.
        """
    train_dataset = tf.keras.utils.image_dataset_from_directory(training_directory, image_size=(imag_height, imag_width),
                                                                batch_size=15500)
    train_data = np.array([])
    train_labels = np.array([])
    # Getting the train data and test data
    for x, y in train_dataset:
        train_data = x.numpy()
        train_labels = y.numpy()
    # searching for a good learning rate
    for learning_rate in [0.001, 0.01, 0.1]:
        kfold = KFold(n_splits=splits_number, shuffle=True)
        kfold_split = kfold.split(train_data, train_labels)
        # iterating through the different splits
        for curr_train, curr_test in kfold_split:
            model = get_model(learning_rate)
            # training the model; using fewer epochs because of how much time training takes for all learning rates
            model.fit(train_data[curr_train], train_labels[curr_train], batch_size=batch_size, epochs=7)
            # getting the evaluation results
            results = model.evaluate(train_data[curr_test], train_labels[curr_test])
            # printing the loss and accuracy
            print(f'Learning rate {learning_rate}: Loss - {results[0]}, Accuracy - {results[1]}')
            # we are only interested in a single validation in this case
            break

# Each of these functions serves a different purpose and should be commented or uncommented as needed
train_and_predict()
# n_fold_cross_validation()
# grid_search_learning_rate()
