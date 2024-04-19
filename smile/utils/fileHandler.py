import os


def list_files():
    smile_files = os.listdir('../data/smile')
    non_smile_files = os.listdir('../data/non_smile')

    return (['../data/smile/' + smile for smile in smile_files]
            , ['../data/non_smile/' + non_smile for non_smile in non_smile_files])


def make_save_directory(nof, fs, ps, en, bs):
    dir_name = ('nof-' + str(nof) + '__fs-' + str(fs)
               + '__ps-' + str(ps) + '__en-' + str(en)
               + '__bs-' + str(bs))
    path = './saves/' + dir_name
    os.mkdir(path)

    return path
