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


def train(model, train_data, train_labels, test_data, test_labels, epoch_num):
    return (
        model.fit(
            train_data,
            to_categorical(train_labels),
            epochs=epoch_num,
            validation_data=(test_data, to_categorical(test_labels)),
        ))


def check_predictions(model, test_data, test_labels):
    example_size = 25
    indexes = random.sample(range(len(test_data)), example_size)

    predictions = model.predict(test_data[indexes])
    real_labels = [test_labels[idx] for idx in indexes]

    incorrect = 0

    for i in range(example_size):
        idx = indexes[i]
        pred = np.argmax(predictions[i])
        real = real_labels[i]
        print(f'{i:n}: predicted: {pred:n} | real: {real:n}')
        imageHandler.show_with_prediction(test_data[idx], predictions[i],
                                          pred, real)
        if pred != real:
            incorrect += 1

    print(f'incorrect predictions: {incorrect:n}/{example_size:n}')
