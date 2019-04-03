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
    print("Data arranged")

    ''' ------------------------- DATA Collected --------------------------- '''

    ''' ------------------------- Starting Neural Network --------------------------- '''

    '''
    num_classes = 8
    
    
    def convolutional_neural_network():
        weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                   'W_fc': tf.Variable(tf.random_normal([125*125*64, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, num_classes]))}
    
        biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([num_classes]))}
    '''

    print("Neural network starts")

    # train_x = train_x.reshape([-1, 500, 500, 1])
    # test_x = test_x.reshape([-1, 500, 500, 1])
    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    train_x = np.array(train_x).reshape(len(train_x), 2048)
    train_y = np.array(train_y)
    test_x = np.array(test_x).reshape(len(test_x), 2048)
    test_y = np.array(test_y)
    print(type(train_x))

    print("Neural Network starts")
    out = input_data(shape=[None, 2048], name='input')

    out = fully_connected(out, 2048, activation='relu')
    convnet = dropout(out, 0.8)

    out = fully_connected(out, 1024, activation='relu')
    out = dropout(out, 0.8)

    out = fully_connected(out, 256, activation='relu')
    out = dropout(out, 0.8)

    out = fully_connected(out, 8, activation='softmax')
    out = regression(out, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(out, tensorboard_verbose=3, tensorboard_dir='log_transfer')

    # model.load('dogs_vs_cats_tflearn.model')
    print("Model training")

    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=50,
              validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, batch_size=16, run_id='dogs_vs_cats')
    print("Model trained")

    model.save('dogs_vs_cats_tflearn_inceptionv3.model')