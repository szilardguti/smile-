import os

import numpy as np


def list_files():
    smile_files = os.listdir('../data/smile')
    non_smile_files = os.listdir('../data/non_smile')

    return (['../data/smile/' + smile for smile in smile_files]
            , ['../data/non_smile/' + non_smile for non_smile in non_smile_files])


def make_save_directory(nof, fs, ps, en, bs):
    dir_name = ('nof-' + str(nof) + '__fs-' + str(fs)
                + '__ps-' + str(ps) + '__en-' + str(en)
                + '__bs-' + str(bs))
    save_path = './saves/' + dir_name

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return save_path


def save_model_data(history, path):
    f = open(path + '/model-data.txt', 'w')
    f.write(f'last training data accuracy: {history.history["accuracy"][-1]}\n')
    f.write(f'best training data accuracy: {max(history.history["accuracy"])} - '
            f'index: {np.argmax(history.history["accuracy"])}\n')

    f.write(f'\nlast test data accuracy: {history.history["val_accuracy"][-1]}\n')
    f.write(f'best test data accuracy: {max(history.history["val_accuracy"])} - '
            f'index: {np.argmax(history.history["val_accuracy"])}\n')

    f.write(f'\nlast training data loss: {history.history["loss"][-1]}\n')
    f.write(f'best training data loss: {min(history.history["loss"])} - '
            f'index: {np.argmin(history.history["loss"])}\n')

    f.write(f'\nlast test data loss: {history.history["val_loss"][-1]}\n')
    f.write(f'best test data loss: {min(history.history["val_loss"])} - '
            f'index: {np.argmin(history.history["val_loss"])}\n')
