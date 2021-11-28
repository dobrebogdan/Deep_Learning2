from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras import metrics

# 1. Load the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
x_train = x_train[20:]
y_train = y_train[20:]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

maxlen = 1000

# 2. Pad sequences

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Set parameters
max_features = 10000
batch_size = 8
embedding_dims = 100
filters = 128
ks = [3, 5, 5]  # kernel_size
hidden_dims = 128
epochs = 10


def get_cnn_model_v1():
    # 3.1. Create the model, no layers yet
    new_mode = Sequential()

    # 3.2. Add the layers (check out the work done in the previous lab)

    ########################################################################################
    # 3.2.1. Add an embedding layer which maps our vocab indices into embedding_dims dimensions
    new_mode.add(Embedding(max_features, embedding_dims, input_length=maxlen))

    # 3.2.2. Dropout with a probability of 0.4
    new_mode.add(Dropout(0.4))

    # 3.2.3. Add a Convolution1D layer, with 128 filters, kernel size ks[0], padding same, activation relu and stride 1
    new_mode.add(Conv1D(filters, ks[0], padding='same', activation='relu', strides=1))

    # 3.2.4. Use max pooling after the CONV layer
    new_mode.add(MaxPooling1D())

    # 3.2.5. Add a CONV layer, similar in properties to what we have above (3.2.3.) and kernel size 5
    new_mode.add(Conv1D(filters, ks[1], padding='same', activation='relu', strides=1))

    # 3.2.6. Add a Batch Normalization layer in order to reduce overfitting
    new_mode.add(BatchNormalization())

    # 3.2.7. Use max pooling again
    new_mode.add(MaxPooling1D())

    # 3.2.8. Add a flatten layer
    new_mode.add(Flatten())

    # 3.2.9. Add a dense layer with hidden_dims hidden units and activation relu
    new_mode.add(Dense(hidden_dims, activation='relu'))

    # 3.2.10. Add a dropout layer with a dropout probability of 0.5
    new_mode.add(Dropout(0.5))

    # 3.2.11. We project onto a single unit output layer, and squash it with a sigmoid
    new_mode.add(Dense(1, activation='sigmoid'))
    ##################################################################################

    # 3.3. Compile the model
    new_mode.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    new_mode.summary()

    return new_mode


model = get_cnn_model_v1()

# 3.4. Train (fit) the model
print(x_test)
print(y_test)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1,
                    shuffle=True)

# 4.1. Evaluate the accuracy and loss on the training set
loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))

# 4.2. Evaluate the accuracy and loss on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


