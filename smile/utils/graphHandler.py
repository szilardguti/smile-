import matplotlib.pyplot as plt


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
