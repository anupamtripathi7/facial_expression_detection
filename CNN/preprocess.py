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


def label_image(contents):  # gives label to each image
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


def label_image_jaffe(contents):  # gives label to each image
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
        print("Error")


def crop_face(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rects = detector(image, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = image[ny:ny + nr, nx:nx + nr]
        # lastimg = cv2.resize(faceimg, (227, 227))
        return faceimg


def get_ck_data(image_directory, label_directory):  # takes images as input by traversing through directories
    count1, count2 = 0, 0
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
                            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                            # img = crop_face(img)
                            cv2.equalizeHist(img)
                            img = cv2.resize(img, (128, 128))
                            if counter <= 2:
                                count2 += 1
                                data.append([np.array(img), np.array([0, 0, 0, 0, 0, 0, 0, 1])])
                            elif int(len(os.listdir(directory2))*0.66) < counter:
                                count2 += 1
                                data.append([np.array(img), np.array(label)])
                else:
                    pass
    return data


def get_jaffe_data(jaffe_directory):
    data = []

    # Append into ck data
    # data = np.load('data_26-3-18_after_crop.npy').tolist()

    count3 = 0
    for image in os.listdir(jaffe_directory):
        if not image.startswith('.'):
            if not image.startswith('R'):
                name = image
                path = os.path.join(jaffe_directory, image)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image = crop_face(image)
                cv2.equalizeHist(image)
                image = cv2.resize(image, (128, 128))
                data.append([np.array(image), np.array(label_image_jaffe(name))])
                count3 += 1
                print(name)

    shuffle(data)
    print("Data shuffled")
    print("Total number of jaffe images are", count3)
    # np.save('data_26-3-18_after_crop_plus_jaffe.npy', data)
    return data
