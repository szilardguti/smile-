# -*- coding: utf-8 -*-
import cv2

from utils import dataLoader, modelTrainer
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

    # TODO: hdft ??


    # TODO: train CNN with keras
    print("\ntrain CNN model...")
    modelTrainer.create_cnn_model(train_data, train_labels)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
