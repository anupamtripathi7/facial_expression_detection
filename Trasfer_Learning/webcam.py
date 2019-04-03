import cv2
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from imutils import face_utils
import numpy as np


model.load('tflearn_expression_detection_new_jaffe.model')


camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)


def get_image():
    retval, im = camera.read()
    return im


for i in range(ramp_frames):
    temp = get_image()
print("Taking image...")
camera_capture = get_image()
file = r'C:/Users/Dell/Downloads/testimg.png'
cv2.imwrite(file, camera_capture)

del camera

file = r'C:/Users/Dell/Downloads/YM.SU1.58.tiff'
img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (500, 500))
print(img)
print(model.predict([np.array(get_coordinates(img)).reshape(None, 136, 1, 1)]))
print(test_y[5])