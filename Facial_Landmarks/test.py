from imutils import face_utils
import numpy as np
import imutils
import dlib
import os
import cv2
from random import shuffle
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


image_directory = r'C:\Users\Dell\Downloads\jaffedbase\jaffe'
label_directory = r'C:\Users\Dell\Downloads\ck+\Emotion'
count1, count2 = 0, 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Dell/Downloads/facial-landmarks/facial-landmarks/shape_predictor_68_face_landmarks.dat")


convnet = input_data(shape=[None, 136, 1, 1], name='input')

convnet = fully_connected(convnet, 2048, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.00001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='log')


# print("Model training")
# model.fit({'input': train_x}, {'targets': train_y}, n_epoch=1200,
#           validation_set=({'input': test_x}, {'targets': test_y}),
#           snapshot_epoch=True, show_metric=True, run_id='cohn_kanade')
# print("Model trained")
#
# # model.save('tflearn_expression_detection_new_jaffe.model')



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
