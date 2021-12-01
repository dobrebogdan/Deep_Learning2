# Submission accuracy: 0.21
import csv
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

def load_training_samples():
    train_data = []
    train_labels = []
    with open('train.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_path = f"train/{row[0]}"
            image = np.array(cv2.imread(image_path)).flatten()
            train_data.append(image)
            train_labels.append(float(row[1]))
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    return (train_data, train_labels)


def load_test_samples():
    test_data = []
    test_ids = []
    with open('test.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_path = f"test/{row[0]}"
            image = np.array(cv2.imread(image_path)).flatten()
            test_data.append(image)
            test_ids.append(row[0])
    test_data = np.array(test_data)
    return (test_data, test_ids)


(train_data, train_labels) = load_training_samples()
print("Done reading!")

classifier = KNeighborsClassifier()
classifier.fit(train_data, train_labels)
(test_data, test_ids) = load_test_samples()
print(test_data)
test_labels = classifier.predict(test_data[0:])

with open('output.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['id', 'label'])
    for i in range(0, len(test_ids)):
        writer.writerow((test_ids[i], int(test_labels[i])))
