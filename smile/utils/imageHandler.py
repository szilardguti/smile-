import math
import random
from typing import Any

import cv2
import matplotlib.pyplot as plt
from cv2 import Mat
from numpy import ndarray, dtype, generic

from utils import exampleHelper


# images from: https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data/data


def read_image_open(path: str, wait: bool = True) -> None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.imshow("color", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    cv2.imshow("resized", resized)

    if wait:
        cv2.waitKey(0)


def read_process_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return resized


# does NOT handle image numbers not divisible by images_col_count
def show_example(data, labels, example_size=100, images_col_count=10, border_size=10) -> None:
    example_images, example_labels, _ = exampleHelper.get_random_examples(data, labels, example_size)

    row_count = math.ceil(len(example_images) / images_col_count)

    rows = [[] for x in range(int(row_count))]
    last_row = 0

    nonsmile_color = [0, 0, 0]
    smile_color = [255, 255, 255]

    for i in range(len(example_images)):
        if len(example_labels) == 0:
            current_color = [255, 255, 255]
        else:
            if example_labels[i]:
                current_color = smile_color
            else:
                current_color = nonsmile_color

        # copy image with colored border
        border_img = cv2.copyMakeBorder(example_images[i].copy(), border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT, value=current_color)
        rows[last_row].append(border_img)
        if (i + 1) % images_col_count == 0:
            last_row += 1

    row_images = []
    for j in range(row_count):
        row_images.append(cv2.hconcat(rows[j]))

    result = cv2.vconcat([row for row in row_images])

    plt.figure(figsize=(10, 10))
    cv2.namedWindow("example", cv2.WINDOW_NORMAL)

    cv2.imshow('example', result)


def show_with_prediction(image, percentage, prediction, real):
    image = image + 0.5
    text = f'%: {percentage[0]*100:.3f}, {percentage[1]*100:.3f} | pred: {prediction:n} | real: {real:n}'

    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.imshow(text, image)
