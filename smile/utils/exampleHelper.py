import random


def get_random_examples(data, labels, example_size):
    example_images = []
    example_labels = []
    example_indexes = random.sample(range(len(data)), example_size)

    for index in example_indexes:
        example_images.append(data[index])
        example_labels.append(labels[index])

    return example_images, example_labels, example_indexes
