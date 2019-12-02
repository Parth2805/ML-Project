import sys

import cifar10 as cifar10
from pip._vendor.distlib.compat import raw_input

import regression

from src import classification


class main():

    def start(self):
        user_response = raw_input("Press 1 to get preloaded models and 2 to run the classification and regression \n")

        if user_response.__eq__('1'):
            print("Running Preloaded models\n")
            cifar10.Cifar10(sys.argv[1], user_response)
        else:
            print("Running classifier and Regression tests\n")

            classifier = classification.class_classification()
            classifier.run_classifier()
            #
            regressor = regression.class_regression()
            regressor.get_regressor()
            # cifar10.Cifar10(sys.argv[1], user_response)


start1 = main()
start1.start()
