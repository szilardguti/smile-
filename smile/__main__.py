# -*- coding: utf-8 -*-
import cv2
import numpy as np

from utils import dataLoader, modelTrainer, dataTransformer
from utils.makeLabels import make_labels
import utils.imageHandler as imageHandler
from utils.listFiles import list_files

import random


def main():
    gray_scale = True

    print("retrieve and load images...")
    (smile_files, non_smile_files) = list_files()
    imageHandler.read_image_open(random.choice(smile_files), False)

    (positive_labels, negative_labels) = make_labels(smile_files, non_smile_files)

    smile_img = [imageHandler.read_process_image(img, gray_scale) for img in smile_files]
    non_smile_img = [imageHandler.read_process_image(img, gray_scale) for img in non_smile_files]

    (train_data, train_labels, test_data, test_labels) \
        = dataLoader.load_data(smile_img, non_smile_img, positive_labels, negative_labels)

    print("image loading is done!")
    imageHandler.show_example(train_data, train_labels)


    print("\ntransform data...")
    train_data = dataTransformer.transform(train_data)
    test_data = dataTransformer.transform(test_data)
    print("transforming data is done!")

    # TODO: hyperparameter loop + graphs

    print("\ntrain CNN model...")
    model = modelTrainer.create_cnn_model()
    modelTrainer.train(model, train_data, train_labels, test_data, test_labels)
    print("model training is done!")

    # save model
    # model.save_weights('cnn.h5')

    print("\ncheck predictions...")
    modelTrainer.checkPredicitons(model, test_data, test_labels)
    print("prediction checking is done!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
