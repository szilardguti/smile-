# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
#
import numpy as np
# import pandas as pd
# from keras.callbacks import EarlyStopping
# from keras.datasets import cifar10
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D
# from keras.optimizers import Adam
# from keras.layers.pooling import MaxPooling2D
# from keras.utils import to_categorical

import os

def create_cnn_model(train_data, train_labels):
    np_train_data = np.array(list(x for x in train_data))

    print("training data shape:", np_train_data.shape)

    #https://victorzhou.com/blog/keras-cnn-tutorial/