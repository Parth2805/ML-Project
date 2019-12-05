import sys

import cifar10 as cifar10
import classification
import regression
from cifar10 import CNN
from pip._vendor.distlib.compat import raw_input

# import regression
from src import classification, regression


class main():

    def start(self):
        user_response = raw_input("Press 1 to get preloaded models and 2 to run the classification and regression \n")
        classifier = classification.class_classification()
        classifier.run_classifier(user_response)
        # regressor = regression.class_regression()
        # regressor.get_regressor()
        # cifar10.Cifar10(sys.argv[1], user_response)
start1 = main()
start1.start()
