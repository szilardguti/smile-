
def make_labels(smile_files, non_smile_files):
    positive_labels = [1 for i in range(len(smile_files))]
    negative_labels = [0 for i in range(len(non_smile_files))]

    return positive_labels, negative_labels
