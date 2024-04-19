import numpy as np


# (batch_size, height, width, channels) - gray channel = 1
def transform(data):
    np_data = np.array(list(x for x in data))

    # normalize data to [-0.5, 0.5]
    np_data = (np_data / 255) - 0.5

    # reshape data for keras
    np_data = np.expand_dims(np_data, axis=3)

    print("new data shape:", np_data.shape)

    return np_data
