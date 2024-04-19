import matplotlib.pyplot as plt


def show(history, key1, key2, title, ylabel, xlabel, legend):
    plt.plot(history.history[key1])
    plt.plot(history.history[key2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.show()
