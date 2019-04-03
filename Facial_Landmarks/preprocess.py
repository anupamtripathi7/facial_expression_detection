import os
import cv2
import numpy as np
from random import shuffle
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from imutils import face_utils
import numpy as np
import imutils
import dlib


def label_ck_image(contents):  # gives label to each image
    label, _ = contents.split(".")
    if label == '   1':
        return [1, 0, 0, 0, 0, 0, 0, 0]  # angry
    elif label == '   2':
        return [0, 1, 0, 0, 0, 0, 0, 0]  # contempt
    elif label == '   3':
        return [0, 0, 1, 0, 0, 0, 0, 0]  # disgust
    elif label == '   4':
        return [0, 0, 0, 1, 0, 0, 0, 0]  # fear
    elif label == '   5':
        return [0, 0, 0, 0, 1, 0, 0, 0]  # happy
    elif label == '   6':
        return [0, 0, 0, 0, 0, 1, 0, 0]  # sad
    elif label == '   7':
        return [0, 0, 0, 0, 0, 0, 1, 0]  # surprise


def label_jaffe_image(contents):  # gives label to each image
    if 'AN' in contents:
        return [1, 0, 0, 0, 0, 0, 0, 0]  # angry
    elif 'NE' in contents:
        return [0, 0, 0, 0, 0, 0, 0, 1]  # contempt
    elif 'DI' in contents:
        return [0, 0, 1, 0, 0, 0, 0, 0]  # disgust
    elif 'FE' in contents:
        return [0, 0, 0, 1, 0, 0, 0, 0]  # fear
    elif 'HA' in contents:
        return [0, 0, 0, 0, 1, 0, 0, 0]  # happy
    elif 'SA' in contents:
        return [0, 0, 0, 0, 0, 1, 0, 0]  # sad
    elif 'SU' in contents:
        return [0, 0, 0, 0, 0, 0, 1, 0]  # surprise
    else:
        print("GADBAD!!!!!!")


def get_coordinates(image):
    rects = detector(image, 1)
    print(enumerate(rects))
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        shape = np.array(shape).reshape(136, 1)
    return shape


def get_jaffe_data(image_directory):  # takes images as input by traversing through directories
    count1, count2 = 0
    data = np.load("data.npy").tolist()

    for img in os.listdir(image_directory):
        if not img.startswith('.'):
            if not img.startswith('R'):
                count1 += 1
                name = img
                path = os.path.join(image_directory, img)
                img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (500, 500))
                count2 += 1
                data.append([get_coordinates(img), np.array(label_image(name))])

    shuffle(data)
    print("Data shuffled")
    print("Total number of images are", count1)
    print("Total number of images chosen are", count2)
    # np.save('data_jaffe.npy', data)
    return data


def get_ck_data(image_directory, label_directory):  # takes images as input by traversing through directories
    count1, count2 = 0

    data = []
    for main_folder in os.listdir(image_directory):
        directory1 = image_directory + "\\" + main_folder
        print("Retrieving data from ", main_folder)
        for sub_folder in os.listdir(directory1):
            if sub_folder.startswith('.'):
                pass
            else:
                directory2 = directory1 + "\\" + sub_folder
            # label_path = label_directory + "\\" + main_folder + "\\" + sub_folder
        label_path = os.path.join(label_directory, main_folder, sub_folder)
        if os.path.exists(label_path):
            for label in os.listdir(label_path):
                if label == "":  # no label, neutral face
                    label = [0, 0, 0, 0, 0, 0, 0, 1]
                else:
                    label_path += "\\" + label
                    f = open(label_path, "r")
                    contents = f.read()
                    label = label_image(contents)  # selecting label according to contents of text file
            counter = 0
            for img in os.listdir(directory2):
                if not img.startswith('.'):
                    counter += 1
                    count1 += 1
                    path = os.path.join(directory2, img)
                    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (500, 500))
                    if counter <= 2:
                        count2 += 1
                        data.append([get_coordinates(img), np.array([0, 0, 0, 0, 0, 0, 0, 1])])
                    elif int(len(os.listdir(directory2)) * 0.66) < counter:
                        count2 += 1
                        data.append([np.array(img), np.array(label)])
        else:
            pass


    shuffle(data)
    print("Data shuffled")
    print("Total number of images are", count1)
    print("Total number of images chosen are", count2)
    # np.save('data.npy', data)
    return data


