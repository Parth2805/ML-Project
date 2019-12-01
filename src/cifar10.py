import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import plot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

N_CHANNELS = 3
SIZE = 32
MODEL_NAME = "RAJANCIFAR10.t7"
EPOCHS = 10
TOLERANCE = 1e-4
device = torch.device('cpu')
DEMO_PATH = "../Results For Demo"


class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # use flatten & softmax the final predictions as done in lab 9
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


def unpickle(file):
    with open(file, 'rb') as file:
        dictionary = pickle.load(file, encoding='bytes')
    return dictionary


def train_model(path):
    print(os.path.isdir(path))
    batch_1 = unpickle(path + "/data_batch_1")
    batch_2 = unpickle(path + "/data_batch_2")
    batch_3 = unpickle(path + "/data_batch_3")
    batch_4 = unpickle(path + "/data_batch_4")
    batch_5 = unpickle(path + "/data_batch_5")
    test_batch = unpickle(path + "/test_batch")
    label_names = unpickle(path + "/batches.meta")

    train_data = np.vstack(
        (batch_1[b'data'], batch_2[b'data'], batch_3[b'data'], batch_4[b'data'], batch_5[b'data']))
    train_labels = np.vstack(
        (batch_1[b'labels'], batch_2[b'labels'], batch_3[b'labels'], batch_4[b'labels'], batch_5[b'labels']))
    train_labels = train_labels.ravel()
    test_data = np.array(list(test_batch[b'data']))
    test_labels = np.array(list(test_batch[b'labels']))
    labels = label_names[b'label_names']
    labels = np.where(labels == b'airplane', 'airplane', labels)
    print(labels)
    training_data = train_data.reshape(train_data.shape[0], N_CHANNELS, SIZE, SIZE)
    testing_data = test_data.reshape(test_data.shape[0], N_CHANNELS, SIZE, SIZE)
    lb = preprocessing.LabelBinarizer()
    training_labels = lb.fit_transform(train_labels)
    testing_labels = lb.fit_transform(test_labels)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data,
                                                                                          training_labels,
                                                                                          test_size=0.1,
                                                                                          random_state=0)
    net = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    torch_train_data = torch.tensor(training_data, dtype=torch.float32)
    torch_train_labels = torch.tensor(training_labels, dtype=torch.int64)
    torch_val_data = torch.tensor(validation_data, dtype=torch.float32)
    torch_val_labels = torch.tensor(validation_labels, dtype=torch.int64)
    torch_test_data = torch.tensor(testing_data, dtype=torch.float32)
    torch_test_labels = torch.tensor(testing_labels, dtype=torch.int64)
    batch_size = 128
    train(net, torch_train_data, torch_train_labels, validation_data, validation_labels, batch_size, optimizer, loss_fn)


def fwd_pass(X, y, optimizer, loss_fn, net, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)
    loss = loss_fn(outputs, torch.max(y, 1)[1])

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss


def test(X, y, optimizer, loss_fn, net, batch_size=500):
    avg_val_acc = 0
    avg_val_loss = 0
    avg_by = X.shape[0] / batch_size

    for i in range(0, X.shape[0], batch_size):
        val_X, val_y = X[i:i + batch_size], y[i:i + batch_size]
        with torch.no_grad():
            val_acc, val_loss = fwd_pass(val_X, val_y, optimizer, loss_fn, net)
            avg_val_acc += val_acc / avg_by
            avg_val_loss += val_loss / avg_by
            print("val_acc {0} and val_loss {1}".format(val_acc, val_loss))
    return avg_val_acc, avg_val_loss


def train(net, torch_train_data, torch_train_labels, torch_val_data, torch_val_labels, batch_size, optimizer, loss_fn):
    prev_val_loss = 1
    training_acc = []
    training_loss = []
    validation_acc = []
    validation_loss = []
    for epoch in range(1, EPOCHS + 1):
        for i in range(0, torch_train_data.shape[0], batch_size):
            X_train = torch_train_data[i:i + batch_size]
            y_train = torch_train_labels[i:i + batch_size]

            train_acc, train_loss = fwd_pass(X_train, y_train, optimizer, loss_fn, net, train=True)
            if i % 50 == 0:
                val_acc, val_loss = test(torch_val_data, torch_val_labels, optimizer, loss_fn, net)
                validation_acc.append(val_acc)
                validation_loss.append(val_loss)
                training_acc.append(train_acc)
                training_loss.append(train_loss)
                print("Validation accuracy: {0} | Validation loss: {1}".format(val_acc * 100, val_loss))

                if prev_val_loss - val_loss > TOLERANCE:
                    print("Val_Loss Decreased from {0} to {1}".format(prev_val_loss, val_loss))
                    prev_val_loss = val_loss
                    print('==> Saving model ...')
                    state = {
                        'net': net,
                        'epoch': epoch,
                        'state_dict': net.state_dict()
                    }
                torch.save(state, DEMO_PATH + "/" + MODEL_NAME)

            print("Iteration: {0} | Loss: {1} | Training accuracy: {2}".format(epoch, train_loss, train_acc * 100))

    print("Finished Training ...")
    # plot.plot(training_acc, training_loss, validation_acc, validation_loss)


class Cifar10:
    def __init__(self, path, load_pretrained_model):
        self.path = path
        if load_pretrained_model is "Y":
            print("load")
            loaded_model = torch.load("../Pretrained Models/" + MODEL_NAME, map_location=device)
            print(loaded_model['net'])
        else:
            train_model(self.path)
