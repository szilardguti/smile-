import os


def list_files():
    smile_files = os.listdir('../data/smile')
    non_smile_files = os.listdir('../data/non_smile')

    return (['../data/smile/' + smile for smile in smile_files]
            , ['../data/non_smile/' + non_smile for non_smile in non_smile_files])
