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
from preprocess import label_image, get_ck_data, crop_face


image_directory = r'C:\Users\Dell\Downloads\ck+\cohn-kanade-images'
label_directory = r'C:\Users\Dell\Downloads\ck+\Emotion'

data = get_ck_data(image_directory, label_directory)
shuffle(data)
np.save('data_ck.npy', data)
