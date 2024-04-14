import random

import numpy as np
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

from smile.utils import exampleHelper
from smile.utils import imageHandler


# https://victorzhou.com/blog/keras-cnn-tutorial/
def create_cnn_model():
    # hyperparameters
    number_of_filters = 8
    filter_size = 3
    pooling_size = 2

    model = Sequential(
        [
            Input(shape=(32, 32, 1)),
            Conv2D(number_of_filters, filter_size, ),
            MaxPooling2D(pool_size=pooling_size),
            Flatten(),
            Dense(2, activation='softmax'),  # probability of the classification
        ]
    )

    model.compile(
        'adam',
        loss='binary_crossentropy',  # bc we use softmax
        metrics=['accuracy'],
    )

    return model


def train(model, train_data, train_labels, test_data, test_labels):
    model.fit(
        train_data,
        to_categorical(train_labels),
        epochs=5,
        validation_data=(test_data, to_categorical(test_labels)),
    )


def check_predictions(model, test_data, test_labels):
    example_size = 25
    indexes = random.sample(range(len(test_data)), example_size)

    predictions = model.predict(test_data[indexes])
    real_labels = [test_labels[idx] for idx in indexes]

    for i in range(example_size):
        idx = indexes[i]
        print(f'{i:n}: predicted: {np.argmax(predictions[i]):n} | real: {test_labels[i]:n}')
        imageHandler.show_with_prediction(test_data[idx], predictions[i],
                                          np.argmax(predictions[i]), real_labels[i])
