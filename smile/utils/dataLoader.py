import math
import random


def load_data(smile_img, non_smile_img, positive_labels, negative_labels):
    percentage = 0.7  # to divide into train or test

    smile_count = math.floor(len(smile_img) * percentage)
    non_smile_count = math.floor(len(non_smile_img) * percentage)

    smile_indexes_train = random.sample(range(len(smile_img)), smile_count)
    smile_indexes_test = set(range(len(smile_img))) - set(smile_indexes_train)

    non_smile_indexes_train = random.sample(range(len(non_smile_img)), non_smile_count)
    non_smile_indexes_test = set(range(len(non_smile_img))) - set(non_smile_indexes_train)

    train_data = []
    train_labels = []
    for sit in smile_indexes_train:
        train_data.append(smile_img[sit])
        train_labels.append(positive_labels[sit])

    for nsit in non_smile_indexes_train:
        train_data.append(non_smile_img[nsit])
        train_labels.append(negative_labels[nsit])

    test_data = []
    test_labels = []
    for sits in smile_indexes_test:
        test_data.append(smile_img[sits])
        test_labels.append(positive_labels[sits])

    for nsits in non_smile_indexes_test:
        test_data.append(non_smile_img[nsits])
        test_labels.append(negative_labels[nsits])

    return (train_data, train_labels, test_data, test_labels)
