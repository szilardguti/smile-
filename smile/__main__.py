# -*- coding: utf-8 -*-
import random

import cv2
import keras
from matplotlib import pyplot as plt

import utils.imageHandler as imageHandler
from smile.utils import graphHandler, fileHandler, cameraHandler
from utils import dataLoader, modelTrainer, dataTransformer
from utils.makeLabels import make_labels


def main():
    print("retrieve and load images...")
    (smile_files, non_smile_files) = fileHandler.list_files()
    imageHandler.read_image_open(random.choice(smile_files), False)

    (positive_labels, negative_labels) = make_labels(smile_files, non_smile_files)

    smile_img = [imageHandler.read_process_image(img) for img in smile_files]
    non_smile_img = [imageHandler.read_process_image(img) for img in non_smile_files]

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
    model = modelTrainer.create_cnn_model(number_of_filters, filter_size, pooling_size)
    history = modelTrainer.train(model, train_data, train_labels, test_data, test_labels, epoch_num, batch_size)
    print("model training is done!")

    print("\nsave model...")
    path = fileHandler.make_save_directory(number_of_filters, filter_size, pooling_size, epoch_num, batch_size)
    model.save(path + '/cnn.keras')
    fileHandler.save_model_data(history, path)
    save_input = input('Use this model for the camera? (y/n): ')
    if save_input == 'y':
        model.save('./camera_model/cnn.keras', overwrite=True)
        print("This model is now used for camera()")

    print(f'Model is saved to folder: {path}')
    print("save model is done!")

    print("\nshow filters...")
    # TODO: show convolutional filters
    graphHandler.show_filters(model)
    cv2.waitKey(0)

    print("show filters is done!")

    print("\nshow graphs...")
    graphHandler.show(history, 'accuracy', 'val_accuracy', 'model accuracy', 'accuracy', 'epoch',
                      ['train', 'test'], 1, save_path=path)
    graphHandler.show(history, 'loss', 'val_loss', 'model loss', 'loss', 'epoch',
                      ['train', 'test'], 2, save_path=path)
    print("show graphs is done...")

    print("\ncheck predictions...")
    modelTrainer.check_predictions(model, test_data, test_labels)
    print("prediction checking is done!")

    plt.show()
    plt.pause(0.01)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def camera():
    # TODO: run camera real time and predict smile
    model = keras.models.load_model('camera_model/cnn.keras')
    model.summary()
    cameraHandler.predict_live_cam(model)


def list_param():
    print("Hyperparams:")
    print(f"Number of Filters: {number_of_filters}")
    print(f"Filter Size: {filter_size}")
    print(f"Pooling Size: {pooling_size}")
    print(f"Epoch Number: {epoch_num}")
    print(f"Batch Size: {batch_size}")


def set_params():
    print("Choose a property to set:")
    print("1. Number of Filters")
    print("2. Filter Size")
    print("3. Pooling Size")
    print("4. Epoch Number")
    print("5. Batch Size")
    print("6. Exit")

    global number_of_filters
    global filter_size
    global pooling_size
    global epoch_num
    global batch_size

    while True:
        set_choice = input("\nEnter your choice (6 to quit): ")

        if set_choice == '1':
            number_of_filters = int(input("Enter the number of filters: "))
            print(f"Number of Filters set to: {number_of_filters}")
        elif set_choice == '2':
            filter_size = int(input("Enter the filter size: "))
            print(f"Filter Size set to: {filter_size}")
        elif set_choice == '3':
            pooling_size = int(input("Enter the pooling size: "))
            print(f"Pooling Size set to: {pooling_size}")
        elif set_choice == '4':
            epoch_num = int(input("Enter the epoch number: "))
            print(f"Epoch Number set to: {epoch_num}")
        elif set_choice == '5':
            batch_size = int(input("Enter the batch size: "))
            print(f"Batch Size set to: {batch_size}")
        elif set_choice == '6':
            print("Exiting property setter...")
            break
        else:
            print("Invalid choice. Please choose a number between 1 and 6.")


if __name__ == '__main__':
    # hyperparameters
    number_of_filters = 8
    filter_size = 3
    pooling_size = 2

    epoch_num = 20
    batch_size = 32

    while True:
        print("Enter 1 to run main(), 2 to run camera(), 3 to list hyperparams, 4 to set hyperparams, or 'q' to quit: ")
        choice = input("Input: ")
        print()
        if choice == '1':
            main()
        elif choice == '2':
            camera()
        elif choice == '3':
            list_param()
        elif choice == '4':
            set_params()
        elif choice.lower() == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid input. Please try again.")
        print("\n")
