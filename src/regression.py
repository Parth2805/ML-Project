import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import plot
import scipy
import scipy.stats as Stats
import sklearn.ensemble as Ensemble
import sklearn.gaussian_process as Gaussian
import sklearn.linear_model as linear
import sklearn.metrics as metrics
import sklearn.model_selection as model_select
import sklearn.neighbors as Neighbors
import sklearn.neural_network as NN
import sklearn.preprocessing as Preprocessing
import sklearn.svm as SVM
import sklearn.tree as Tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.neighbors
import sklearn.neural_network
from scipy.stats import reciprocal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import class_weight
from sklearn.utils import resample

RESULTS_FOR_DEMO = "../Results For Demo/"
DATASETS = "../Datasets/"
PRETRAINED_MODEL = "../Pretrained Models/"


class class_regression:
    '''Contains all the regression logic'''

    def grid_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test, name, cv=5):
        print("Grid Search CV {0}".format(name))
        model = model_select.GridSearchCV(classifier, param_grid, cv=cv, verbose=1, scoring="r2").fit(X_train, y_train)
        print("Grid Search CV")
        print("Best Estimator: ", model.best_estimator_)
        print("Mean Squared Error: ",
              metrics.mean_squared_error(y_test, model.best_estimator_.predict(X_test)))
        print("R2 Score: ", metrics.r2_score(y_test, model.best_estimator_.predict(X_test)))
        pickle.dump(model.best_estimator_, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
        pickle.dump(model.best_params_, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
        plot.plot_learning_curve(model.best_estimator_, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=cv)

    def random_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test, name, cv=5, n_iter=30):
        print("Random Search {0}".format(name))
        model = model_select.RandomizedSearchCV(classifier, param_grid, cv=cv, n_iter=n_iter, verbose=1,
                                                random_state=0, scoring="r2").fit(X_train, y_train)
        print("Random Search CV")
        print("Best Estimator: ", model.best_estimator_)
        print("Mean Squared Error: ",
              metrics.mean_squared_error(y_test, model.best_estimator_.predict(X_test)))
        print("R2 Score: ", metrics.r2_score(y_test, model.best_estimator_.predict(X_test)))
        pickle.dump(model.best_estimator_, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
        pickle.dump(model.best_params_, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
        plot.plot_learning_curve(model.best_estimator_, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=cv)

    def load_pretrained_models(self, name, X_test, y_test):
        print("Loading PreTrained model: ", name)
        model = pickle.load(open(PRETRAINED_MODEL + name + ".sav", 'rb'))
        print("Mean Squared Error: ",
              metrics.mean_squared_error(y_test, model.predict(X_test)))
        print("R2 Score: ", metrics.r2_score(y_test, model.predict(X_test)))


    def merck_model(self, act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, name, pretrained=False):
        print(name)
        print(pretrained)
        act2_predict = act2.predict(X_test_act2)
        act4_predict = act4.predict(X_test_act4)
        act2_r2 = metrics.r2_score(y_test_act2, act2_predict)
        act4_r2 = metrics.r2_score(y_test_act4, act4_predict)

        print("R2 Score: act2 {0}: & act4: {1} ".format(act2_r2, act4_r2))
        print("Average R2 Score: ", (act2_r2 + act4_r2) / 2)

        pearson_act2 = Stats.pearsonr(act2_predict, y_test_act2)
        pearson_act4 = Stats.pearsonr(act4_predict, y_test_act4)
        print("PearsonR: act2 {0}: & act4: {1} ".format(pearson_act2, pearson_act4))
        if pretrained:
            print("Pretrained Model")
        else:
            pickle.dump(act2, open(RESULTS_FOR_DEMO + name + "BestModelACT2.sav", 'wb'))
            pickle.dump(act2.get_params, open(RESULTS_FOR_DEMO + name + "BestParamsACT2.sav", 'wb'))
            pickle.dump(act4, open(RESULTS_FOR_DEMO + name + "BestModelACT4.sav", 'wb'))
            pickle.dump(act4.get_params, open(RESULTS_FOR_DEMO + name + "BestParamsACT4.sav", 'wb'))

    def get_regressor(self, userResponse):
        print('Running regressors for the following datasets: \n')
        self.WineQuality(userResponse)
        # self.Communities_Crime()
        # self.QSAR_aquatic_toxicity()
        # self.Parkinson_Speech()
        self.Facebook_metrics(userResponse)
        # self.Bike_Sharing()
        # self.Student_Performance()
        # self.Concrete_Compressive_Strength()
        # self.SGEMM_GPU_kernel_performance()
        # self.Merck_Molecular_Activity_Challenge()

    def WineQuality(self, userResponse):
        print('Running Regression for 1.WineQuality dataset')
        DATASET_NAME = "WineQuality/"
        '''
        ### **Preprocessing**
        '''
        file = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(file, sep=';')
        data = pd.DataFrame(df)
        data = data.values
        file = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        df2 = pd.read_csv(file, sep=';')
        data2 = pd.DataFrame(df2)
        data2 = data2.values
        data = np.vstack((data, data2))
        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:11], data[:, 11], test_size=0.20, random_state=0)

        'Function for NAN'
        # print(data.shape)
        list = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (not (np.isfinite(data[i][j]))):
                    print("There is a nan at:", i, j)
                else:
                    list.append(data[i][j])
        # print(len(list) / 12)

        if (userResponse == '2'):
            '''
            ### **Linear Regression**
            '''
            lr = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
            print("Mean Squared Error: ",metrics.mean_squared_error(y_test, lr.predict(X_test)))
            print("R2 Score: ", metrics.r2_score(y_test, lr.predict(X_test)))
            name="WineQuality_Linear_Regression"
            pickle.dump(lr, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(lr.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(lr, name + " Learning Curve", X_train, y_train, (0.5, 1.01),cv=5)


            '''
            ### **SVR**
            '''
            svm = sklearn.svm.SVR()

            param = {"kernel": np.array(["rbf"]),
                     "degree": np.array([1, 2, 3]),
                     "C": np.array([1, 2, 5, 10, 20]),
                     "gamma": np.array([0.1, 1, 5, 10])}

            self.random_search_cv(svm, param, X_train, y_train, X_test, y_test, "WineQuality_SVR")
            '''
            ### **Decision Tree**
            '''
            dt = sklearn.tree.DecisionTreeRegressor(random_state=0)

            param = {'max_depth': np.arange(1, 15, 1),
                     'splitter': ['best', 'random'],
                     'max_features': np.arange(1, 11, 1),
                     'min_samples_split': np.arange(2, 20, 1)}
            self.grid_search_cv(dt, param, X_train, y_train, X_test, y_test, "WineQuality_Decision_Tree")
            '''
            ### **Random Forest**
            '''

            rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'max_features': np.array([1, 2, 5, 11]),
                     'min_samples_split': np.array([2, 3, 5])}

            self.grid_search_cv(rf, param, X_train, y_train, X_test, y_test, "WineQuality_Random_Forest")
            '''
            ## **Ada Boost**
            '''

            ada = sklearn.ensemble.AdaBoostRegressor(random_state=0)

            param = dict(n_estimators=np.arange(50, 250, 10),
                         loss=['linear', 'square']
                         )
            self.grid_search_cv(ada, param, X_train, y_train, X_test, y_test, "WineQuality_Ada_Boost")
            '''
            ## **Neural Network**
            '''
            mlp = sklearn.neural_network.MLPRegressor(activation='relu', n_iter_no_change=10, momentum=0.9,
                                                      learning_rate='adaptive', random_state=0, verbose=True,
                                                      warm_start=True, early_stopping=True, )
            param = {
                "solver": np.array(['adam']),
                "learning_rate_init": np.arange(0.001, 0.1),
                "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
            }

            self.grid_search_cv(mlp, param, X_train, y_train, X_test, y_test, "WineQuality_Neural_Network")
            '''
            ## **Gaussian Process**
            '''
            gp = sklearn.gaussian_process.GaussianProcessRegressor(random_state=0, normalize_y=True, alpha=0.005).fit(
                X_train, y_train)
            # print(gp.score(X_train, y_train))
            # print(gp.score(X_test, y_test))
            name="WineQuality_Gaussian_Process"
            print("Mean Squared Error: ",
                  metrics.mean_squared_error(y_test, gp.predict(X_test)))
            print("R2 Score: ", metrics.r2_score(y_test, gp.predict(X_test)))
            pickle.dump(gp, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(gp.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(gp, name + " Learning Curve", X_train, y_train, (0.5, 1.01),cv=5)

        else:

            self.load_pretrained_models("WineQuality_SVRModel",X_test, y_test)
            self.load_pretrained_models("WineQuality_Linear_RegressionModel", X_test, y_test)
            self.load_pretrained_models("WineQuality_Decision_TreeModel", X_test, y_test)
            self.load_pretrained_models("WineQuality_Random_ForestModel", X_test, y_test)
            self.load_pretrained_models("WineQuality_Ada_BoostModel", X_test, y_test)
            self.load_pretrained_models("WineQuality_Neural_NetworkModel", X_test, y_test)
            self.load_pretrained_models("WineQuality_Gaussian_ProcessModel", X_test, y_test)

    def Communities_Crime(self):
        print('Running Regression for 2.Communities_Crime dataset')

    def QSAR_aquatic_toxicity(self):
        print('Running Regression for 3.QSAR_aquatic_toxicity dataset')
        file = "http://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv"
        df = pd.read_csv(file, sep=';', header=None)
        data = pd.DataFrame(df)
        data = data.astype(float)

        X = data.loc[:, :7]
        y = data.loc[:, 8]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        '''LINEAR REGRESSION'''

        '''SVR'''

        param_grid = {"kernel": ['linear', 'rbf'],
                      "C": np.logspace(0, 3, 4),
                      "gamma": np.logspace(-2, 1, 4)}

        '''DECISION TREE REGRESSOR'''

        param_grid = {'max_depth': np.arange(1, 20, 2),
                      'splitter': ['best', 'random']}

        '''RANDOM FOREST REGRESSOR'''

        param_grid = {'max_depth': np.arange(1, 20, 1),
                      'min_samples_split': np.array([2, 3, 5])}

        '''ADABOOST REGRESSOR'''

        param_grid = {
            'n_estimators': np.arange(50, 250, 10),
            'loss': ['linear', 'square']
        }

        '''GAUSSIAN PROCESS REGRESSOR'''

        param_grid = {
            "alpha": [1e-10, 1e-9, 1e-8, 1e-5]
        }

        '''NEURAL NETWORK REGRESSOR'''

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.001, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
        }

    def Parkinson_Speech(self):
        print('Running Regression for 4.Parkinson_Speech dataset')

    def Facebook_metrics(self, userResponse):
        print('Running Regression for 5.Facebook_metrics dataset')

        '''
        ### **Preprocessing**
        '''

        file = "../Datasets/5_Facebook.csv"
        df = pd.read_csv(file, sep=';')

        le = sklearn.preprocessing.LabelEncoder().fit(df.iloc[:, 1])
        df.iloc[:, 1] = le.transform(df.iloc[:, 1])

        df.fillna(0, axis=0, inplace=True)
        data = df.values
        print(data)
        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:18], data[:, 18], test_size=0.20, random_state=0)

        'Function for NAN'
        list = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (np.isnan(data[i][j])):
                    print("There is a nan at:", i, j)
                else:
                    list.append(data[i][j])
        # print(len(list) / 19)

        if (userResponse == '2'):
            '''
            ### **Linear Regression**
            '''

            lr = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
            print("Mean Squared Error: ", metrics.mean_squared_error(y_test, lr.predict(X_test)))
            print("R2 Score: ", metrics.r2_score(y_test, lr.predict(X_test)))
            name = "Facebook_Linear_Regression"
            pickle.dump(lr, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(lr.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(lr, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=5)

            '''
            ### **SVR**
            '''

            svm = sklearn.svm.SVR()
            param = {
                "kernel": ["linear"],
                "C": np.arange(0.01, 2)
            }
            self.grid_search_cv(svm, param, X_train, y_train, X_test, y_test, "Facebook_SVR")

            '''
            ### **Decision Tree**
            '''
            dt = sklearn.tree.DecisionTreeRegressor(random_state=0)

            param = {'max_depth': np.arange(1, 18, 1),
                     'splitter': ['best', 'random'],
                     'max_features': np.arange(1, 18, 1),
                     'min_samples_split': np.arange(2, 20, 1)}
            self.grid_search_cv(dt, param, X_train, y_train, X_test, y_test, "Facebook_Decision_Tree")

            '''
            ### **Random Forest**
            '''
            rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'max_features': np.array([1, 2, 5, 10, 15, 18]),
                     'min_samples_split': np.array([2, 3, 5])}

            self.grid_search_cv(rf, param, X_train, y_train, X_test, y_test, "Facebook_Random_Forest")
            '''
            ## **Ada Boost**
            '''

            ada = sklearn.ensemble.AdaBoostRegressor(random_state=0)

            param = dict(n_estimators=np.arange(50, 250, 10),
                         loss=['linear', 'square']
                         )

            self.grid_search_cv(ada, param, X_train, y_train, X_test, y_test, "Facebook_Ada_Boost")

            '''
            ## **Neural Network**
            '''

            mlp = sklearn.neural_network.MLPRegressor(activation='relu', n_iter_no_change=10, momentum=0.9,
                                                      learning_rate='adaptive', random_state=0, verbose=True,
                                                      warm_start=True, early_stopping=True, )

            param = {
                "solver": ['adam'],
                "learning_rate_init": np.arange(0.001, 0.1),
                "hidden_layer_sizes": [(128, 64, 32, 16), (32, 16, 8), (64, 32, 16)]
            }
            self.grid_search_cv(mlp, param, X_train, y_train, X_test, y_test, "Facebook_Neural_Network")
            '''
            ## **Guassian Process**
            '''
            gp = sklearn.gaussian_process.GaussianProcessRegressor()

            param = {'alpha': np.arange(0, 1, 0.001),
                     'normalize_y': ['True', 'False']}
            self.grid_search_cv(gp, param, X_train, y_train, X_test, y_test, "Facebook_Gaussian_Process")
        else:

            self.load_pretrained_models("Facebook_SVRModel", X_test, y_test)
            self.load_pretrained_models("Facebook_Linear_RegressionModel", X_test, y_test)
            self.load_pretrained_models("Facebook_Decision_TreeModel", X_test, y_test)
            self.load_pretrained_models("Facebook_Random_ForestModel", X_test, y_test)
            self.load_pretrained_models("Facebook_Ada_BoostModel", X_test, y_test)
            self.load_pretrained_models("Facebook_Neural_NetworkModel", X_test, y_test)
            self.load_pretrained_models("Facebook_Gaussian_ProcessModel", X_test, y_test)

    def Bike_Sharing(self):
        print('Running Regression for 6.Bike_Sharing dataset')

    def Student_Performance(self):
        print('Running Regression for 7.Student_Performance dataset')

        df = pd.read_csv("../Datasets/student-por.csv", sep=';')

        X = df.loc[:, 'school':'G2']
        y = df.loc[:, 'G3']
        X = X.replace(
            {'GP': 0, 'MS': 1, 'F': 1, 'M': 0, 'U': 0, 'R': 1, 'LE3': 0, 'GT3': 1, 'A': 0, 'T': 1, 'yes': 1, 'no': 0,
             'father': 0, 'mother': 1, 'other': 2, 'teacher': 0, 'at_home': 1, 'health': 3, 'services': 4, 'home': 0,
             'reputation': 1, 'course': 3}).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        '''LINEAR REGRESSION'''

        '''SVR'''

        param_grid = {"kernel": ['linear', 'rbf'],
                      "C": np.logspace(0, 3, 4),
                      "gamma": np.logspace(-2, 1, 4)}

        '''DECISION TREE REGRESSOR'''

        param_grid = {'max_depth': np.arange(1, 20, 2),
                      'splitter': ['best', 'random']}

        '''RANDOM FOREST REGRESSOR'''

        param_grid = {'max_depth': np.arange(1, 20, 1),
                      'min_samples_split': np.array([2, 3, 5])}

        '''ADABOOST REGRESSOR'''

        param_grid = {
            'n_estimators': np.arange(50, 250, 10),
            'loss': ['linear', 'square']
        }

        '''GAUSSIAN PROCESS REGRESSOR'''

        param_grid = {
            "alpha": [1e-10, 1e-9, 1e-8, 1e-5]
        }

        '''NEURAL NETWORK REGRESSOR'''

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.001, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
        }

    def Concrete_Compressive_Strength(self, userResponse):
        print('Running Regression for 8.Concrete_Compressive_Strength dataset')
        df = pd.read_excel(DATASETS + "Concrete_Data.xls")
        df.columns = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate",
                      "FineAggregate", "Age", " Concrete Strength"]
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
        scaler = Preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled)

        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled)
        if userResponse is "2":
            # SVM
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1e-3, 1e-2, 1e-1, 10]
                          }
            self.grid_search_cv(SVM.SVR(kernel='rbf'), param_grid, X_train_scaled, y_train, X_test_scaled, y_test,
                                "ConcreteSVM")

            # LR
            lr_model = linear.LinearRegression().fit(X_train_scaled, y_train)
            print("Linear Regression Mean Squared Error: ",
                  metrics.mean_squared_error(y_test, lr_model.predict(X_test_scaled)))
            print("Linear Regression R2 Score: ", metrics.r2_score(y_test, lr_model.predict(X_test_scaled)))
            filename = "ConcreteLRModel.sav"
            pickle.dump(lr_model, open(RESULTS_FOR_DEMO + filename, 'wb'))
            filename1 = "ConcreteLRBestParam.sav"
            pickle.dump(lr_model.get_params, open(RESULTS_FOR_DEMO + filename1, 'wb'))

            # DTC
            params_grid = {'max_depth': np.arange(1, 10),
                           'splitter': ['best', 'random'],
                           'max_features': np.arange(1, 9, 1),
                           'min_samples_split': np.arange(2, 10, 1)}

            self.grid_search_cv(Tree.DecisionTreeRegressor(random_state=0), params_grid, X_train_scaled, y_train,
                                X_test_scaled, y_test, "ConcreteDTC")

            # RFC
            params_grid = {'max_depth': np.arange(1, 10),
                           'max_features': np.arange(1, 9),
                           'min_samples_split': np.arange(2, 10),
                           'n_estimators': [estimator for estimator in (2 ** i for i in range(0, 8))]}
            self.random_search_cv(Ensemble.RandomForestRegressor(random_state=0), params_grid, X_train_scaled, y_train,
                                  X_test, y_test, "ConcreteRFC")

            # MLP
            params_grid = {
                "activation": ['relu', 'tanh'],
                "solver": ['adam'],
                "learning_rate_init": Stats.reciprocal(0.001, 0.1),
                "hidden_layer_sizes": [(128, 64, 32, 16), (32, 16, 8), (64, 32, 16)]
            }
            mlp = NN.MLPRegressor(n_iter_no_change=10, momentum=0.9, learning_rate='adaptive', random_state=0,
                                  verbose=True, warm_start=True, early_stopping=True)
            # Check for error_score = np.nan
            self.random_search_cv(mlp, params_grid, X_train_scaled, y_train, X_test, y_test, "ConcreteMLP")

            # Adaboost
            params_grid = {"n_estimators": [estimator for estimator in (2 ** i for i in range(0, 8))],
                           "loss": ['linear', 'square'],
                           "learning_rate": [0.01, 0.1, 1]
                           }
            self.grid_search_cv(Ensemble.AdaBoostRegressor(random_state=0), params_grid, X_train_scaled, y_train,
                                X_test_scaled, y_test, "ConcreteAdaboost")

            # GPR
            params_grid = {
                "alpha": [1e-10, 1e-9, 1e-8]
            }
            self.grid_search_cv(Gaussian.GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", random_state=0),
                                params_grid, X_train_scaled, y_train, X_test_scaled, y_test, "ConcreteDTC")
        else:
            # SVM
            self.load_pretrained_models("SVR_concrete_model", X_train_scaled, y_train, X_test_scaled, y_test, 5)

            # DTC
            self.load_pretrained_models("decsion_concrete_tree_model", X_train_scaled, y_train, X_test_scaled, y_test,
                                        5)

            # RFC
            self.load_pretrained_models("random_concrete_forest_model", X_train_scaled, y_train, X_test_scaled, y_test,
                                        5)

            # LR
            self.load_pretrained_models("Linear_regression_model", X_train_scaled, y_train, X_test_scaled, y_test, 5)

            # Adaboost
            self.load_pretrained_models("AdaBoost_concrete_model", X_train_scaled, y_train, X_test_scaled, y_test, 5)

            # GNB
            self.load_pretrained_models("gaussian_concrete_model", X_train_scaled, y_train, X_test_scaled, y_test, 5)

            # MLP
            self.load_pretrained_models("neural_network_concrete_model", X_train_scaled, y_train, X_test_scaled, y_test,
                                        5)

    def SGEMM_GPU_kernel_performance(self,userResponse):
        print('Running Regression for 9.SGEMM_GPU_kernel_performance dataset')

        '''
        ### **Preprocessing**
        '''
        file = "/content/drive/My Drive/Regression/9_Sgemm_GPU/sgemm_product.csv"
        df = pd.read_csv(file, sep=',')
        data = df.values
        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:14], data[:, 14:15], test_size=0.20,
                                                            random_state=0)

        # scaler=StandardScaler().fit(X_train)
        # X_train=scaler.transform(X_train)
        # X_test=scaler.transform(X_test)
        # y_train=y_train.astype(int)
        # y_test=y_test.astype(int)
        y_test = np.ravel(y_test)
        y_train = np.ravel(y_train)

        # 'Function for NAN'
        # # print(data.shape)
        # list=[]
        # for i in range(data.shape[0]):
        #   for j in range(data.shape[1]):
        #     if(np.isnan(data[i][j])):
        #       print("There is a nan at:",i,j)
        #     else:
        #       list.append(data[i][j])
        # print(len(list)/18)

        if(userResponse=='2'):
            # %%
            '''
            ### **Linear Regression**
            '''
            lr = sklearn.linear_model.LinearRegression().fit(X_train, y_train)

            print("Mean Squared Error: ", metrics.mean_squared_error(y_test, lr.predict(X_test)))
            print("R2 Score: ", metrics.r2_score(y_test, lr.predict(X_test)))
            name = "(SGEMM_Linear_Regression"
            pickle.dump(lr, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(lr.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(lr, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=5)

            '''
            ### **SVR**
            '''

            # # %%
            # svm = sklearn.svm.SVR(kernel="rbf").fit(X_train, y_train)
            #
            # print(svm.score(X_train, y_train))
            # print(svm.score(X_test, y_test))


            '''
            ### **Decision Tree**
            '''

            dt = sklearn.tree.DecisionTreeRegressor(random_state=0)

            param = {'max_depth': np.arange(1, 10, 1),
                     'splitter': ['best', 'random'],
                     'max_features': np.arange(1, 10, 1),
                     'min_samples_split': np.arange(2, 13, 1)}

            self.random_search_cv(dt,param,X_train,y_train,X_test,y_test,"SGEMM_Decision_Tree")


            '''
            ### **Random Forest**
            '''

            # %%
            rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'max_features': np.array([1, 2, 3, 5, 7, 9]),
                     'min_samples_split': np.array([2, 3, 5])}

            self.random_search_cv(rf, param, X_train, y_train, X_test, y_test,"SGEMM_Random_Forest")
            '''
            ## **Ada Boost**
            '''

            # %%
            ada = sklearn.ensemble.AdaBoostRegressor(random_state=0)

            param = dict(n_estimators=np.arange(50, 250, 10),
                         loss=['linear', 'square']
                         )

            self.grid_search_cv(ada, param, X_train, y_train, X_test, y_test,"SGEMM_Ada_Boost")

            # %%
            '''
            ## **Neural Network**
            '''

            # %%
            mlp = sklearn.neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(128, 64, 32, 16),
                                                      activation='relu', n_iter_no_change=10, momentum=0.9,
                                                      learning_rate='adaptive', random_state=0, verbose=True,
                                                      warm_start=True, early_stopping=True, ).fit(X_train, y_train)

            print("Mean Squared Error: ", metrics.mean_squared_error(y_test, mlp.predict(X_test)))
            print("R2 Score: ", metrics.r2_score(y_test, mlp.predict(X_test)))
            name = "SGEMM_Neural_Network"
            pickle.dump(mlp, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(mlp.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(lr, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=5)
            '''
            ## **Gaussian Process**
            '''
            #
            # # %%
            # gp = sklearn.gaussian_process.GaussianProcessRegressor(random_state=0, normalize_y=True, alpha=0.05).fit(
            #     X_train, y_train)
            # print(gp.score(X_train, y_train))
            # print(gp.score(X_test, y_test))
        else:
            print("hello")




    def Merck_Molecular_Activity_Challenge(self, userResponse):
        print('Running Regression for 10.Merck_Molecular_Activity_Challenge dataset')
        act2 = np.load(DATASETS + "ac2_cache.npz")
        # print(act2.files)
        act4 = np.load(DATASETS + "ac4_cache.npz")

        X_act2, y_act2 = act2['arr_0'], act2['arr_1']
        X_act4, y_act4 = act4['arr_0'], act4['arr_1']
        X_train_act2, X_test_act2, y_train_act2, y_test_act2 = train_test_split(X_act2, y_act2, test_size=0.2,
                                                                                random_state=0)
        X_train_act4, X_test_act4, y_train_act4, y_test_act4 = train_test_split(X_act4, y_act4, test_size=0.2,
                                                                                random_state=0)
        if userResponse is "2":
            # LR
            lr_model_act2 = linear.LinearRegression().fit(X_train_act2, y_train_act2)
            print("act2 done")
            lr_model_act4 = linear.LinearRegression().fit(X_train_act4, y_train_act4)

            self.merck_model(lr_model_act2, lr_model_act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "LR")

            # SVR
            svr_act2 = SVM.SVR().fit(X_train_act2, y_train_act2)
            svr_act4 = SVM.SVR().fit(X_train_act4, y_train_act4)
            self.merck_model(svr_act2, svr_act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "SVR")

            # DTC
            dct_act2 = Tree.DecisionTreeRegressor(random_state=0).fit(X_train_act2, y_train_act2)
            dct_act4 = Tree.DecisionTreeRegressor(random_state=0).fit(X_train_act4, y_train_act4)
            self.merck_model(dct_act2, dct_act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "DTC")

            # RFC
            rfr_act2 = Ensemble.RandomForestRegressor(random_state=0).fit(X_train_act2, y_train_act2)
            rfr_act4 = Ensemble.RandomForestRegressor(random_state=0).fit(X_train_act4, y_train_act4)
            self.merck_model(rfr_act2, rfr_act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "RFC")

            # Adaboost
            ada_act2 = Ensemble.AdaBoostRegressor(random_state=0).fit(X_train_act2, y_train_act2)
            ada_act4 = Ensemble.AdaBoostRegressor(random_state=0).fit(X_train_act4, y_train_act4)
            self.merck_model(ada_act2, ada_act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "Adaboost")

            # GPR
            gau_act2 = Gaussian.GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", random_state=0).fit(X_train_act2,
                                                                                                        y_train_act2)
            gau_act4 = Gaussian.GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", random_state=0).fit(X_train_act4,
                                                                                                        y_train_act4)
            self.merck_model(gau_act2, gau_act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "GPR")

            # MLP
            param_grid = {
                "activation": ['relu', 'tanh'],
                "learning_rate_init": Stats.reciprocal(0.001, 0.1),
                "hidden_layer_sizes": [(512, 256, 128, 64, 32), (128, 64, 32, 16), (64, 32, 16)]
            }

            mlp_act2 = NN.MLPRegressor(solver='adam', n_iter_no_change=10, momentum=0.9, learning_rate='adaptive',
                                       random_state=0, verbose=True, warm_start=True, early_stopping=True)
            nn_act2 = model_select.RandomizedSearchCV(mlp_act2, param_grid, cv=5, scoring='r2', verbose=5,
                                                      error_score='np.nan').fit(X_train_act2, y_train_act2)
            print("act2 done")
            mlp_act4 = NN.MLPRegressor(solver='adam', n_iter_no_change=10, momentum=0.9, learning_rate='adaptive',
                                       random_state=0, verbose=True, warm_start=True, early_stopping=True)
            nn_act4 = RandomizedSearchCV(mlp_act4, param_grid, cv=5, scoring='r2', verbose=5, error_score='np.nan').fit(
                X_train_act4, y_train_act4)
            self.merck_model(nn_act2.best_estimator_, nn_act4.best_estimator_, X_test_act2, y_test_act2, X_test_act4,
                             y_test_act4, "MLP")
        else:
            act2 = pickle.load(open(PRETRAINED_MODEL + "Linear_act2_regression_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "Linear_act4_regression_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "LR", pretrained=True)

            act2 = pickle.load(open(PRETRAINED_MODEL + "svr_act2_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "svr_act4_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "SVR", pretrained=True)

            act2 = pickle.load(open(PRETRAINED_MODEL + "dct_act2_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "dct_act4_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "DTC", pretrained=True)

            act2 = pickle.load(open(PRETRAINED_MODEL + "rfr_act2_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "rfr_act4_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "RFC", pretrained=True)

            act2 = pickle.load(open(PRETRAINED_MODEL + "ada_act2_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "ada_act4_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "Adaboost",
                             pretrained=True)

            act2 = pickle.load(open(PRETRAINED_MODEL + "gau_act2_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "gau_act4_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "GPR", pretrained=True)

            act2 = pickle.load(open(PRETRAINED_MODEL + "mlp_act2_random_model.sav", "rb"))
            act4 = pickle.load(open(PRETRAINED_MODEL + "mlp_act4_random_model.sav", "rb"))
            self.merck_model(act2, act4, X_test_act2, y_test_act2, X_test_act4, y_test_act4, "MLP", pretrained=True)
