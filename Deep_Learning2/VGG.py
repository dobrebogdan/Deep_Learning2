# Importing libraries
import csv
from keras.optimizer_v2.adam import Adam
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16

# some constants for the program
batch_size = 32
imag_height = 128
imag_width = 55
training_directory = './train'
classes_number = 5
epochs_number = 15
splits_number = 5

""" 
Loading the dataset from the images directory. I previously split the directory in 5 subdirectories, one for each class.
"""
train_dataset = tf.keras.utils.image_dataset_from_directory(training_directory, image_size=(imag_height, imag_width),
                                                            batch_size=batch_size)

# Getting the classnames
class_names = train_dataset.class_names

# using the AUTOTUNE for the prefetch function to decide the number of prefetched elements at runtime
AUTOTUNE = tf.data.AUTOTUNE

# using a 1000 shuffle buffer
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)


# Function used for getting the model
def get_model(learning_rate=None):
    # The VGG16 model
    vgg_model = VGG16(input_shape=(128, 55, 3), include_top=False, weights='imagenet')
    for layer in vgg_model.layers:
        layer.trainable = False

    # custom layers to be added to the VGG
    custom_layers = [
        # Rescailing image
        layers.Rescaling(1. / 255, input_shape=(imag_height, imag_width, 3)),
        # Flattening output
        layers.Flatten(),
        # Dense layer
        layers.Dense(1024, activation='relu'),
        # Dropout layer
        layers.Dropout(0.5),
        # Dense layer with 5 nodes
        layers.Dense(classes_number)
    ]

    # passing the output from the VGG through all the custom layer
    output = vgg_model.output
    for layer in custom_layers:
        output = layer(output)

    # Creating the Model
    model = tf.keras.models.Model(vgg_model.input, output)

    # Compiling the model
    optimizer = 'adam'
    # Setting Adam optimizer with the right learning rate if we get one
    if learning_rate:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # Returning the model
    return model


# Training the model and predicting the labels
def train_and_predict():
    model = get_model()
    model.fit(train_dataset, epochs=epochs_number)
    # model.save('./vgg_model')

    # predicting the labels for the test data
    output_data = []
    with open('test.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_path = f"test/{row[0]}"
            # reading image
            img = tf.keras.utils.load_img(image_path, target_size=(imag_height, imag_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # predictions for the image
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            # getting the label for the image
            label = np.argmax(score)
            # adding the row id and the label
            output_data.append([row[0], int(class_names[label])])

    # writing the row id and label for each image to the output file
    with open('output.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'label'])
        for output_row in output_data:
            writer.writerow(output_row)

# 5 fold cross validation
def n_fold_cross_validation():
    """
    Loading the dataset from the images directory. I split the directory in 5 subdirectories, one for each class.
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
    # Using KFold to split in 5 parts for cross validation
    kfold = KFold(n_splits=splits_number, shuffle=True)
    kfold_split = kfold.split(train_data, train_labels)
    step = 0
    accuracies = []
    # iterating through the different splits
    for curr_train, curr_test in kfold_split:
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
        Loading the dataset from the images directory. I split the directory in 5 subdirectories, one for each class.
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

# Each of these functions serve a different purpose and should be commented or uncommented as needed
train_and_predict()
# n_fold_cross_validation()
# grid_search_learning_rate()
