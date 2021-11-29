def partial_train():
    pass


def evaluate():
    pass

def full_train():
    pass


import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


files_and_labels = []
with open('train.csv') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        image = cv2.imread(f"train/{row[0]}") / 255
        label = row[1]
        files_and_labels.append((image, label))

data_size = len(files_and_labels)
training_data_zie = int(data_size * 0.8)
training_data = files_and_labels[:training_data_zie]
validation_data = files_and_labels[training_data_zie:]
