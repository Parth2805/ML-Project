import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

PATH = "../Results For Demo/"


def plot_training_val_graph(training_acc, training_loss, validation_acc, validation_loss):
    plt.plot(training_acc, label="training_acc")
    plt.plot(validation_acc, label="validation_acc")
    plt.legend();
    plt.savefig(PATH + "cpu_acc_plot.png")
    plt.figure();
    plt.plot(training_loss, label="training_loss")
    plt.plot(validation_loss, label="validation_loss")
    plt.legend();
    plt.savefig(PATH + "cpu_loss_plot.png")
    plt.show(block=True)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    print(title)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(PATH + title + ".png")
