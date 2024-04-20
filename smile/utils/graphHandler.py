import cv2
import matplotlib.pyplot as plt
import numpy as np


def show(history, key1, key2, title, ylabel, xlabel, legend, plot_num, save_path=None):
    plt.figure(plot_num)
    plt.plot(history.history[key1])
    plt.plot(history.history[key2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')

    if save_path:
        plt.savefig(save_path + '/' + title.replace(' ', '-') + '.png')


def show_filters(model):
    conv_layer = model.layers[0]

    kernels, biases = conv_layer.get_weights()
    # Normalize the kernel weights to [0, 255] range for display
    kernels_normalized = ((kernels - np.min(kernels)) / (np.max(kernels) - np.min(kernels))) * 255
    for i in range(kernels.shape[3]):
        kernel_i = kernels_normalized[:, :, :, i]

        # Convert the kernel to uint8 for displaying with cv2
        kernel_i_uint8 = kernel_i.astype(np.uint8)

        # Resize the kernel for better visualization (optional)
        kernel_i_resized = cv2.resize(kernel_i_uint8, (200, 200))

        # Display the kernel using OpenCV
        cv2.imshow(f'Filter-{i + 1}', kernel_i_resized)
