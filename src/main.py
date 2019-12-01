from pip._vendor.distlib.compat import raw_input
import classification
import regression


'''user_response = raw_input("Enter Y for Yes and N for No\n")

if(user_response.__eq__('Y')):
    print("You said Yes")
else:
    print("You said No")'''



class main():

    def start(self):
        user_response = raw_input("Press 1 to get preloaded models and 2 to run the classification and regression \n")

        if(user_response.__eq__('1')):
            print("Running Preloaded models\n")
        else:
            print("Running classifier and Regression tests\n")
            classifier = classification.class_classification()
            classifier.run_classifier()

            regressor = regression.class_regression()
            regressor.get_regressor()



start1 = main()
start1.start()

