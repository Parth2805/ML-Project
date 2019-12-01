import matplotlib.pyplot as plt

PATH = "../Results For Demo/"


def plot_training_val_graph(training_acc, training_loss, validation_acc, validation_loss):
    plt.plot(training_acc, label="training_acc")
    plt.plot(validation_acc, label="validation_acc")
    plt.legend();
    plt.savefig(PATH + "cpu_acc_plot.jpg")
    plt.figure();
    plt.plot(training_loss, label="training_loss")
    plt.plot(validation_loss, label="validation_loss")
    plt.legend();
    plt.savefig(PATH + "cpu_loss_plot.jpg")
    plt.show(block=True)
