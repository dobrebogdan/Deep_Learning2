import csv
import cv2

with open('train.csv') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        image = cv2.imread(f"train/{row[0]}")
        label = row[1]
        cv2.imwrite(f"train/{label}/{row[0]}", image)
