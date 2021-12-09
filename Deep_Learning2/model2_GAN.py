# accuracy close to random chance
# try to use stuff from other models
import csv
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.optimizer_v2.adam import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras import backend
import tensorflow as tf

img_height = 128
img_width = 55
data_dir = './train'
num_classes = 5
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result


def define_discriminator(in_shape=(128, 165, 1), n_classes=5):
    in_image = Input(shape=in_shape)
    fe = Conv2D(128, (3, 3), strides=(4, 4), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3, 3), strides=(4, 4), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3, 3), strides=(4, 4), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(n_classes)(fe)
    c_out_layer = Activation('softmax')(fe)
    classifier = Model(in_image, c_out_layer)
    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    d_out_layer = Lambda(custom_activation)(fe)
    discriminator = Model(in_image, d_out_layer)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return discriminator, classifier


def define_generator(latent_dim):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 128 * 4 * 3
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((4, 3, 128))(gen)
    gen = Conv2DTranspose(128, (4, 4), strides=(4, 5), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4, 4), strides=(8, 11), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv2D(1, (4, 3), activation='tanh', padding='same')(gen)
    model = Model(in_lat, out_layer)
    return model


def define_gan(generator, discriminator):
    discriminator.trainable = False
    gan_output = discriminator(generator.output)
    model = Model(generator.input, gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model



def load_training_samples():
    train_data = []
    train_labels = []
    train_no = 1000
    with open('train.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            train_no -= 1
            if not train_no:
                break
            image_path = f"train/{row[0]}"
            img = tf.keras.utils.load_img(
                image_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.reshape(img_array, (128, 165))
            train_data.append(img_array)
            train_labels.append(float(row[1]) - 1)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    return (train_data, train_labels)


testX = []
def load_real_samples():
    (trainX, trainy) = load_training_samples()
    print('##')
    print(np.shape(trainX[0]))
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    print(X.shape, trainy.shape)
    return [X, trainy]


def select_supervised_samples(dataset, n_samples=100, n_classes=5):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(0, n_classes):
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)


def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input


def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict(z_input)
    y = zeros((n_samples, 1))
    return images, y


def train(generator, discriminator, classifier, gan, dataset, latent_dim, n_epochs=20, n_batch=100):
    X_sup, y_sup = select_supervised_samples(dataset)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    print('###')
    print(n_steps)
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    for i in range(n_steps):
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = classifier.train_on_batch(Xsup_real, ysup_real)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = discriminator.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
        d_loss2 = discriminator.train_on_batch(X_fake, y_fake)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan.train_on_batch(X_gan, y_gan)
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i + 1, c_loss, c_acc * 100, d_loss1, d_loss2, g_loss))
    return classifier


latent_dim = 100
discriminator, classifier = define_discriminator()
generator = define_generator(latent_dim)
gan = define_gan(generator, discriminator)
dataset = load_real_samples()
classifier = train(generator, discriminator, classifier, gan, dataset, latent_dim)
print("Training done")

output_data = []
predicted_labels = []
img_arrays = []
image_ids = []
with open('test.csv') as file:
    csv_reader = csv.reader(file)
    num_samples = 0
    for row in csv_reader:
        num_samples += 1
        image_path = f"test/{row[0]}"
        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.reshape(img_array, (128, 165)).astype('float32')
        img_array = tf.expand_dims(img_array, -1)
        img_arrays.append(img_array)
        image_ids.append(row[0])
    img_arrays = np.array(img_arrays).reshape((-1, 128, 165, 1))
    predictions = discriminator.predict(img_arrays)
    for i in range(0, num_samples):
        score = tf.nn.softmax(predictions[i])
        label = np.argmax(score)
        output_data.append([image_ids[i], int(label+1)])


print("Predict done")
with open('output.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['id', 'label'])
    for output_row in output_data:
        writer.writerow(output_row)
