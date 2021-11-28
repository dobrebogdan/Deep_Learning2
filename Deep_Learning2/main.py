def partial_train():
    pass


def evaluate():
    pass

def full_train():
    pass

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import csv


batch_size = 32
img_height = 128
img_width = 55
train_ds = tf.keras.utils.image_dataset_from_directory(
  directory='train',
  validation_split=0.2,
  subset="training",
  seed=123,
  #image_size=(img_height, img_width),
  batch_size=batch_size)