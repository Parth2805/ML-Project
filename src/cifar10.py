import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import plot
# import pydotplus as pdp
import sklearn.model_selection as model_select
import sklearn.tree as Tree
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

N_CHANNELS = 3
SIZE = 32
MODEL_NAME = "CIFAR10.t7"
EPOCHS = 10
TOLERANCE = 1e-4
device = torch.device('cpu')
DEMO_PATH = "../Results For Demo/"
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
PRETRAINED_MODEL = "../Pretrained Models/"
TREE_MODEL = "Decision_tree_classifier_cifar10"
RESULTS_FOR_DEMO = "../Results For Demo/"


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


class VanillaBackprop():
    def __init__(self, model, loss_fn):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.loss_fn = loss_fn

    def generate_gradient_first_layer(self, input_image, target_class):
        for i in range(10):
            model_output = self.model(input_image)  # for grad, add to(device)
            self.model.zero_grad()
            label_as_var = torch.from_numpy(np.asarray([target_class], dtype=np.int64))
            loss = self.loss_fn(model_output, label_as_var)
            loss.backward()
            input_image.data += 0.2 * input_image.grad.data
        return input_image


class FGST():
    def __init__(self, model, loss_fn, alpha):
        self.model = model
        self.model.eval()
        self.alpha = alpha
        self.loss_fn = loss_fn

    def adversarial_noise(self, image, label):
        image.grad = None
        model_output = self.model(image.to(device))
        label_as_var = torch.from_numpy(np.asarray([label], dtype=np.int64))
        loss = self.loss_fn(model_output, label_as_var)
        loss.backward()
        return torch.sign(image.grad.data)

    def generate_gradient_on_target_class(self, index, target_class, torch_test_data):
        random_image = torch_test_data[index].reshape(1, 3, 32, 32)
        random_tensor = torch.tensor(random_image, dtype=torch.float32, requires_grad=True)
        perturbations = 0.0
        for i in range(5):
            perturbations += self.adversarial_noise(random_tensor, target_class).numpy()
            perturbations_tensor = torch.tensor(perturbations)
            random_tensor.data = random_tensor.data - perturbations_tensor * self.alpha
        return random_tensor, perturbations_tensor;


def unpickle(file):
    with open(file, 'rb') as file:
        dictionary = pickle.load(file, encoding='bytes')
    return dictionary


def train_model(path, userResponse):
    print("Path Exists: ", os.path.isdir(path))
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
    print("Labels: ", labels)
    training_data = train_data.reshape(train_data.shape[0], N_CHANNELS, SIZE, SIZE)
    testing_data = test_data.reshape(test_data.shape[0], N_CHANNELS, SIZE, SIZE)
    lb = preprocessing.LabelBinarizer()
    training_labels = lb.fit_transform(train_labels)
    testing_labels = lb.fit_transform(test_labels)

    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data,
                                                                                          training_labels,
                                                                                          test_size=0.1, random_state=0)
    torch_train_data = torch.tensor(training_data, dtype=torch.float32)
    torch_train_labels = torch.tensor(training_labels, dtype=torch.int64)
    torch_val_data = torch.tensor(validation_data, dtype=torch.float32)
    torch_val_labels = torch.tensor(validation_labels, dtype=torch.int64)
    torch_test_data = torch.tensor(testing_data, dtype=torch.float32)
    torch_test_labels = torch.tensor(testing_labels, dtype=torch.int64)
    batch_size = 128
    net = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    if userResponse is "2":
        print("training model")
        train(net, torch_train_data, torch_train_labels, torch_val_data, torch_val_labels, batch_size, optimizer,
              loss_fn)
        net = torch.load(RESULTS_FOR_DEMO + MODEL_NAME, map_location=device)
        test_acc, test_loss = test(torch_test_data, torch_test_labels, optimizer, loss_fn, net['net'])
        print("Testing Accuracy {0} | Testing Loss {1}".format(test_acc, test_loss))

        print("Activation Maximization")
        VBP = VanillaBackprop(net, loss_fn)
        activation_maximization(VBP)

        print("FGST")
        fgst(net, testing_data, torch_test_data, loss_fn)
        train_DTC(training_data, train_labels, testing_data, test_labels)

    else:
        print("pretrained models")
        loaded_model = torch.load(PRETRAINED_MODEL + MODEL_NAME, map_location=device)
        # print(loaded_model['net'])
        test_acc, test_loss = test(torch_test_data, torch_test_labels, optimizer, loss_fn, loaded_model['net'])
        print("Testing Accuracy {0} | Testing Accuracy {1}".format(test_acc, test_loss))
        print("Activation Maximization")
        VBP = VanillaBackprop(loaded_model['net'], loss_fn)
        activation_maximization(VBP)

        print("FGST")
        fgst(loaded_model['net'], testing_data, torch_test_data, loss_fn)

        dct_loaded_model = pickle.load(open(PRETRAINED_MODEL + TREE_MODEL + ".sav", 'rb'))
        n_samples, channels, height, width = testing_data.shape
        dct_testing_data = testing_data.reshape(n_samples, channels * height * width)
        tree_pred = dct_loaded_model.score(dct_testing_data, test_labels)
        print("Tree Model Pretrained Accuracy: ", tree_pred)


def train_DTC(training_data, train_labels, testing_data, test_labels):
    n_samples, channels, height, width = training_data.shape
    dct_training_data = training_data.reshape(n_samples, channels * height * width)
    dct_testing_data = testing_data.reshape(n_samples, channels * height * width)
    hyper_params = {'max_depth': np.arange(5, 50, 5),
                    'max_leaf_nodes': np.arange(5, 50, 5),
                    'min_samples_leaf': [5, 10],
                    'min_samples_split': [3, 5, 10]
                    }

    dtree_model = model_select.RandomizedSearchCV(
        Tree.DecisionTreeClassifier(random_state=0, criterion='entropy'), param_distributions=hyper_params,
        verbose=3, cv=3, scoring="accuracy").fit(dct_training_data, train_labels)
    tree_pred = dtree_model.score(dct_testing_data, test_labels)
    print("Tree Model Trained Accuracy: ", tree_pred)

    filename = "Decision_tree_classifier_cifar10.sav"
    pickle.dump(dtree_model.best_estimator_, open(DEMO_PATH + "%s" % filename, 'wb'))
    filename = "Decision_tree_classifier_cifar10_best_params.sav"
    pickle.dump(dtree_model.best_params_, open(DEMO_PATH + "%s" % filename, 'wb'))
    sklearn.tree.plot_tree(dtree_model.best_estimator_, fontsize=12)
    filename = "Decision_tree_plot"
    dot_data = Tree.export_graphviz(dtree_model.best_estimator_, out_file=None, filled=True, rounded=True,
                                    special_characters=True)
    graph = pdp.graph_from_dot_data(dot_data)
    graph.write_pdf(DEMO_PATH + "%s" % filename + ".pdf")


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
            val_acc, val_loss = fwd_pass(val_X.view(batch_size, N_CHANNELS, SIZE, SIZE).to(device), val_y.to(device),
                                         optimizer, loss_fn, net)
            avg_val_acc += val_acc / avg_by
            avg_val_loss += val_loss / avg_by
            print("val/test_acc {0} and val/test_loss {1}".format(val_acc, val_loss))
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
    plot.plot_training_val_graph(training_acc, training_loss, validation_acc, validation_loss)


def activation_maximization(VBP):
    random_image = np.ones(32 * 32 * 3).reshape(3, 32, 32)
    random_tensor_image = random_image.reshape(1, 3, 32, 32)
    f, axarr = plt.subplots(1, 11, figsize=(32, 32));
    axarr[0].imshow(random_image.transpose(1, 2, 0));
    axarr[0].set_xticks([]);
    axarr[0].set_yticks([]);
    axarr[0].set_title("random image");
    j = 0;
    for i in range(10):
        random_tensor = torch.tensor(random_tensor_image, dtype=torch.float32, requires_grad=True)
        input_image_gradients = VBP.generate_gradient_first_layer(random_tensor, i)
        input_image_gradients_numpy = input_image_gradients.data.cpu().numpy()[0]
        gray_vanilla = convert_to_gray_scale(input_image_gradients_numpy)
        axarr[i + 1].imshow(np.repeat(gray_vanilla, 3, axis=0).transpose(1, 2, 0));
        axarr[i + 1].set_xticks([]);
        axarr[i + 1].set_yticks([]);
        axarr[i + 1].set_title("activating \nimage \nfor class\n {0}".format(labels[j]));
        j = j + 1
    plt.show(block=True)
    f.savefig(DEMO_PATH + "activation_maximization.jpg")


def fgst(net, testing_data, torch_test_data, loss_fn):
    # testing to add perturbations to one image acc to target class
    index = 100
    target_class = 5
    fgst1 = FGST(net, loss_fn, alpha=0.5)
    perturbated_result, pertubation_tensor = fgst1.generate_gradient_on_target_class(index, target_class,
                                                                                     torch_test_data)
    model_perturbated_pred = torch.softmax(net(perturbated_result.to(device)), dim=1)
    model_actual_pred = torch.softmax(net(torch_test_data[index].view(-1, 3, 32, 32).to(device)), dim=1)

    result_to_image = perturbated_result.data.cpu().numpy()[0]
    perturbation_tensor_to_image = pertubation_tensor.data.cpu().numpy()[0]

    f, axarr = plt.subplots(1, 3, figsize=(9, 9));

    axarr[0].imshow(testing_data[index].transpose(1, 2, 0))
    axarr[0].set_xticks([]);
    axarr[0].set_yticks([]);
    axarr[0].set_title("""Model Actual Image \nPrediction class: {0}\n accuracy: {1:.2f}% """.
                       format(labels[model_actual_pred.argmax().item()],
                              (model_actual_pred[0][model_actual_pred.argmax().item()] * 100).item()));

    axarr[1].imshow(perturbation_tensor_to_image.transpose(1, 2, 0));
    axarr[1].set_xticks([]);
    axarr[1].set_yticks([]);
    axarr[1].set_title("Perturbation Image for\n target class: {0}".format(labels[target_class]));

    axarr[2].imshow(result_to_image.transpose(1, 2, 0).astype(np.uint8))
    axarr[2].set_xticks([]);
    axarr[2].set_yticks([]);
    axarr[2].set_title("""Model Perturbated Image \nPrediction class: {0}\n accuracy: {1:.2f}% """.
                       format(labels[model_perturbated_pred.argmax().item()],
                              (model_perturbated_pred[0][model_perturbated_pred.argmax().item()] * 100).item()));
    # plt.imsave("perturabted{0}.jpg".format(index),(perturbated_image.transpose(1,2,0)).astype(np.uint8))

    f.savefig(DEMO_PATH + "/fgst.jpg");


def convert_to_gray_scale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


class Cifar10:
    def __init__(self, path, load_pretrained_model):
        self.path = path
        train_model(path, load_pretrained_model)
