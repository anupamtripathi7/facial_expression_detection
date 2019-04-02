import cv2
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from imutils import face_utils
import numpy as np


if __name__ == '__main__':

    data = np.load('data_ck_plus_jaffe.npy')
    training = data[0:-(int(len(data)*0.3))]  # separating training and testing data
    testing = data[-(int(len(data)*0.3)):]
    train_x, train_y, test_x, test_y = [], [], [], []
    for img in range(len(training)):
        train_x.append((training[img][0]))
        train_y.append(training[img][1])
    for img in range(len(testing)):
        test_x.append((testing[img][0]))
        test_y.append(testing[img][1])
    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    train_x = np.array(train_x).reshape(3686, 128, 128, 1)
    train_y = np.array(train_y)
    test_x = np.array(test_x).reshape(1579, 128, 128, 1)
    test_y = np.array(test_y)
    print(type(train_x))


    convnet = input_data(shape=[None, 128, 128, 1], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2)

    convnet = conv_2d(convnet, 256, 3, activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2)

    convnet = fully_connected(convnet, 4096, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2048, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 8, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.00001, loss='categorical_crossentropy', name='targets')


    model = tflearn.DNN(convnet)


    model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='log_cnn')


    # model.load('tflearn_expression_detection.model')
    print("Model training")
    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=100,
              validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, batch_size=32, run_id='cohn_kanade')
    print("Model trained")


    model.save('tflearn_expression_detection.model')
