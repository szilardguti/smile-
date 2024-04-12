import numpy as np
from keras import Input
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

from utils import exampleHelper, imageHandler


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


def checkPredicitons(model, test_data, test_labels):
    _, _, indexes = exampleHelper.get_random_examples(test_data, test_labels, 25)

    predictions = model.predict(test_data[indexes])
    real_labels = [test_labels[idx] for idx in indexes]

    # Print our model's predictions.
    print(predictions)
    # Check our predictions against the ground truths.
    print(real_labels)  # [7, 2, 1, 0, 4]

    imageHandler.show_with_prediction(test_data[0], predictions[0], np.argmax(predictions[0]), real_labels[0])