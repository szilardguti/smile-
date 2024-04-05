
def make_labels(smile_files: list[str], non_smile_files: list[str]) -> (list[int], list[int]):
    positive_labels = [1 for i in range(len(smile_files))]
    negative_labels = [0 for i in range(len(non_smile_files))]

    return positive_labels, negative_labels
