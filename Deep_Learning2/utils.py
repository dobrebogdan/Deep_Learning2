import csv
import cv2

with open('output.csv', 'w') as file:
    spamwriter = csv.writer(file, delimiter=',')
    spamwriter.writerow(['Spam', 2.0])
