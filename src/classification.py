import itertools
import pickle

import numpy as np
import pandas as pd
import plot
import scipy.stats as Stats
import sklearn
import sklearn.ensemble as Ensemble
import sklearn.linear_model  as Linear
import sklearn.metrics as metrics
import sklearn.model_selection as model_select
import sklearn.neighbors as Neighbors
import sklearn.neural_network as NN
import sklearn.preprocessing as Preprocessing
import sklearn.svm
import sklearn.tree as Tree
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from scipy.io import arff
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RESULTS_FOR_DEMO = "../Results For Demo/"
DATASETS = "../Datasets/"
PRETRAINED_MODEL = "../Pretrained Models/"


class class_classification:
    '''Contains all the classifiers'''

    def grid_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test, name, cv=5):
        model = model_select.GridSearchCV(classifier, param_grid, cv=cv, verbose=1).fit(X_train, y_train)
        print("Grid Search CV {0}".format(name))
        print("Best Estimator: ", model.best_estimator_)
        print("Average HyperParameter Search Accuracy: ", model.best_score_)
        print("Testing Accuracy: ", model.best_estimator_.score(X_test, y_test))
        print("Training Accuracy: ", model.best_estimator_.score(X_train, y_train))
        pickle.dump(model.best_estimator_, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
        pickle.dump(model.best_params_, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
        plot.plot_learning_curve(model.best_estimator_, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=cv)

    def random_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test, name, cv=5, n_iter=30):
        model = model_select.RandomizedSearchCV(classifier, param_distributions=param_grid, cv=cv, n_iter=n_iter,
                                                verbose=1, random_state=0).fit(X_train, y_train)
        print("Random Search {0}".format(name))
        print("Best Estimator: ", model.best_estimator_)
        print("Average HyperParameter Search Accuracy: ", model.best_score_)
        print("Testing Accuracy: ", model.best_estimator_.score(X_test, y_test))
        print("Training Accuracy: ", model.best_estimator_.score(X_train, y_train))
        pickle.dump(model.best_estimator_, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
        pickle.dump(model.best_params_, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
        plot.plot_learning_curve(model.best_estimator_, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=cv)

    def load_pretrained_models(self, name):
        print("Loading PreTrained model: ", name)
        model = pickle.load(PRETRAINED_MODEL + name + ".sav")
        print("Testing Accuracy: ", model.score(X_test, y_test))
        print("Training Accuracy: ", model.score(X_train, y_train))
        plot.plot_learning_curve(model, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=cv)

    def run_classifier(self, userResponse):
        print('Running classifiers for the following datasets: \n')
        # self.Diabetic_Retinopathy()
        # self.Default_of_credit_card_clients(userResponse)
        # self.Breast_Cancer_Wisconsin()
        # self.Statlog_Australian()
        # self.Statlog_German()
        self.Steel_Plates_Faults(userResponse)
        # self.Adult()
        # self.Yeast()
        # self.Thoracic_Surgery_Data()
        # self.Seismic_Bumps(userResponse)

    def Diabetic_Retinopathy(self):
        print('Running classification for 1.Diabetic Retinopathy dataset')

        file = DATASETS + "1_DiabeticRetinopathy.arff"
        df, metadata = arff.loadarff(file)

        data = pd.DataFrame(df)
        data = data.values
        data[:, 19] = np.where(data[:, 19] == b'0', 0, data[:, 19])
        data[:, 19] = np.where(data[:, 19] == b'1', 1, data[:, 19])

        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:19], data[:, 19], test_size=0.20, random_state=0)

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        scaler = StandardScaler()
        scaler.fit(X_train[:, 8:18])
        X_train[:, 8:18] = scaler.transform(X_train[:, 8:18])
        X_test[:, 8:18] = scaler.transform(X_test[:, 8:18])

        '''Logistic Regression'''

        lr = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=10000)

        param = {'solver': ["sag", "saga", "liblinear"],
                 'C': [0.1, 0.2, 0.5, 1, 1.5, 2, 5, 7, 10, 12, 15]
                 }

        self.random_search_cv(lr, param, X_train, y_train, X_test, y_test, "DiabeticLogisticRegression", 3)

        # %%
        '''
        ### K-**Neighbors**
        '''
        # %%

        k_n = sklearn.neighbors.KNeighborsClassifier()

        param = {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20, 50, 100, 200, 500]}

        # # model = sklearn.model_selection.RandomizedSearchCV(estimator=k_n, param_distributions=param, cv=5,
        #                                                     random_state=0).fit(X_train, y_train)
        #  print('Best Score: ', model.best_score_)
        #  print('Best Params: ', model.best_params_)
        #
        #  # %%
        #  filename = 'K_Neighbors.sav'
        #  pickle.dump(model, open(filename, 'wb'))
        #  filename = 'K_Neighbors_best_param.sav'
        #  pickle.dump(model.best_params_, open(filename, 'wb'))

        # %%
        '''
        ### **SVM**
        '''
        # %%

        svm = sklearn.svm.SVC(random_state=0)

        param = dict(kernel=['rbf', 'linear'],
                     degree=[1, 2, 3],
                     C=Stats.reciprocal(0.01, 2),
                     gamma=Stats.reciprocal(0.01, 2))

        # model = sklearn.model_selection.RandomizedSearchCV(estimator=svm, param_distributions=param, verbose=10,
        #                                                    cv=5).fit(X_train, y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # filename = 'SVM.sav'
        # pickle.dump(model, open(filename, 'wb'))
        # filename = 'SVM_best_param.sav'
        # pickle.dump(model.best_params_, open(filename, 'wb'))

        # %%
        '''
        ### **Decision Tree**
        '''

        # %%

        dt = sklearn.tree.DecisionTreeClassifier(random_state=0)

        param = {'max_depth': np.arange(1, 20, 1),
                 'splitter': ['best', 'random'],
                 'max_features': np.arange(1, 19, 1),
                 'min_samples_split': np.arange(2, 20, 1)}

        # model = sklearn.model_selection.GridSearchCV(estimator=dt, param_grid=param, verbose=3, cv=5).fit(X_train,
        #                                                                                                   y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # filename = 'Decision_Tree.sav'
        # pickle.dump(model, open(filename, 'wb'))
        # filename = 'Decision_Tree_best_param.sav'
        # pickle.dump(model.best_params_, open(filename, 'wb'))
        #
        # print(model.score(X_train, y_train))
        # print(model.score(X_test, y_test))

        # %%
        '''
        ### **Random Forest**
        '''

        # %%
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=0)

        param = {'max_depth': np.arange(1, 20, 1),
                 'max_features': np.arange(1, 19, 1),
                 'min_samples_split': np.arange(2, 20, 1)}

        # model = sklearn.model_selection.RandomizedSearchCV(estimator=rf, param_distributions=param, verbose=3,
        #                                                    cv=5).fit(X_train, y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # model = sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=0, min_samples_split=6,
        #                                                 max_features=2, max_depth=12).fit(X_train, y_train)
        #
        # # %%
        # print(model.score(X_train, y_train))
        # print(model.score(X_test, y_test))
        #
        # # %%
        # filename = 'Random_Forest.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Random_Forest_best_param.sav'
        # pickle.dump(model.get_params, open(path + filename, 'wb'))

        # %%
        '''
        ### **Gaussian naive Bayes classification**
        '''
        # %%

        gb = sklearn.naive_bayes.GaussianNB().fit(X_train, y_train)

        print(gb.score(X_train, y_train))
        print(gb.score(X_test, y_test))

        # %%
        # filename = 'Guassain_Naives_Bayes.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Guassain_Naives_Bayes_best_param.sav'
        # pickle.dump(model.get_params, open(path + filename, 'wb'))

        # %%
        '''
        ### **Neural Network**
        '''

        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(20, 15, 20, 2), random_state=0,
                                                     max_iter=1000).fit(X_train, y_train)

        # %%
        # mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
        #                                            learning_rate='adaptive', random_state=0, verbose=True,
        #                                            warm_start=True, early_stopping=True)
        #
        # param_grid = {
        #     "solver": ['adam', 'sgd'],
        #     "learning_rate_init": reciprocal(0.001, 0.1),
        #     "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
        # }
        # model = sklearn.model_selection.RandomizedSearchCV(mlp, param_grid, n_iter=30, cv=5, scoring="accuracy",
        #                                                    verbose=5).fit(X_train, y_train)
        #
        # # %%
        # print(model.best_params_)
        # print(model.score(X_test, y_test))
        # print(model.score(X_train, y_train))
        #
        # # %%
        # filename = 'Neural_Network.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Neural_Network_best_param.sav'
        # pickle.dump(model.best_params_, open(path + filename, 'wb'))

        # %%
        '''

        '''

        # %%
        '''
        ### **Ada Boost**
        '''

        # %%
        ada = sklearn.ensemble.AdaBoostClassifier(random_state=0)

        param = dict(n_estimators=np.arange(50, 250, 10),
                     algorithm=['SAMME.R', 'SAMME']
                     )

        # model = sklearn.model_selection.GridSearchCV(estimator=ada, param_grid=param, verbose=10, cv=5).fit(X_train,
        #                                                                                                     y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # print("Validation Score:", model.best_score_)
        # print("Testing Score:", model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'AdaBoostClassifier.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'AdaBoostClassifier_best_param.sav'
        # pickle.dump(model.best_params_, open(path + filename, 'wb'))

        # %%
        '''
        ## **Testing**
        '''

        # %%
        # model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(20, 15, 20, 2), random_state=0,
        #                                              max_iter=1000).fit(X_train, y_train)
        # print(mlp.score(X_train, y_train))
        # print(mlp.score(X_test, y_test))
        #
        # # %%
        # filename = 'Neural_Network.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Neural_Network_best_param.sav'
        # pickle.dump(model.get_params, open(path + filename, 'wb'))
        #
        # # %%
        # filename = 'Neural_Network.sav'
        # model = pickle.load(open(path + filename, 'rb'))
        # print(model.get_params)

    def Default_of_credit_card_clients(self, userResponse):
        if userResponse is "2":
            print('Running classification for 2.Default of credit card clients dataset')

            df = pd.read_excel(DATASETS + "default of credit card clients.xls", skiprows=1)
            del df['ID']
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
            y_train = df_train['default payment next month']
            X_train = df_train
            del X_train['default payment next month']

            y_test = df_test['default payment next month']
            X_test = df_test
            del X_test['default payment next month']
            scaler = Preprocessing.StandardScaler().fit(X_train.iloc[:, 11:23]);
            X_train.loc[:, 11:23] = scaler.transform(X_train.iloc[:, 11:23]);
            X_test.loc[:, 11:23] = scaler.transform(X_test.iloc[:, 11:23]);

            mean_limit_bal = np.mean(X_train['LIMIT_BAL'])
            std_limit_bal = np.std(X_train['LIMIT_BAL'])
            X_train['LIMIT_BAL'] = (X_train['LIMIT_BAL'] - mean_limit_bal) / std_limit_bal;
            X_test['LIMIT_BAL'] = (X_test['LIMIT_BAL'] - mean_limit_bal) / std_limit_bal;

            print(X_train.shape, X_test.shape)
            # SVM
            param_grid = {'C': [0.001, 0.01, 0.1, 1], 'gamma': [0.01, 0.1, 1]}
            svm = sklearn.svm.SVC(kernel="rbf", random_state=0)
            self.grid_search_cv(svm, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardSVM", cv=3)

            # Decision Tree Classifier
            param_grid = {'max_depth': np.arange(4, 10),
                          'max_leaf_nodes': [5, 10, 20, 100],
                          'min_samples_split': [2, 5, 10, 20]
                          }
            dtc = Tree.DecisionTreeClassifier(random_state=0, criterion='gini')
            self.grid_search_cv(dtc, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardDTC", cv=3)

            # RFC
            param_grid = {'max_depth': np.arange(4, 10),
                          'max_leaf_nodes': [5, 10, 20],
                          'min_samples_split': [2, 5],
                          'n_estimators': [estimator for estimator in (2 ** i for i in range(0, 8))]}
            rfc = Ensemble.RandomForestClassifier(random_state=0, criterion='gini')
            self.grid_search_cv(rfc, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardRFC", cv=3)

            # LR
            param_grid = {
                "penalty": ['l2'],
                "C": Stats.reciprocal(0.001, 1000),
                "fit_intercept": [True, False],
                "solver": ['lbfgs', 'sag', 'saga']
            }
            lr = Linear.LogisticRegression(random_state=0)
            self.random_search_cv(lr, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardLR", cv=3)

            # Adaboost
            param_grid = {
                "n_estimators": [50, 70, 100, 120],
                "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
                "algorithm": ["SAMME", "SAMME.R"]
            }
            adaboost = Ensemble.AdaBoostClassifier(random_state=0)
            self.grid_search_cv(adaboost, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardAdaboost",
                                cv=3)

            # KNN
            param_grid = {
                "n_neighbors": [10, 50, 100],
                "weights": ['uniform', 'distance'],
                "leaf_size": [15, 30, 50, 100]
            }
            knn = Neighbors.KNeighborsClassifier()
            self.grid_search_cv(knn, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardKNN", cv=3)

            # GNB
            param_grid = {
                "var_smoothing": [1e-07, 1e-08, 1e-09]
            }
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardGaussian",
                                cv=3)

            # MLP
            param_grid = {
                "solver": ['adam', 'sgd'],
                "learning_rate_init": Stats.reciprocal(0.001, 0.1),
                "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
            }
            nn = NN.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                  learning_rate='adaptive',
                                  random_state=0, verbose=True, warm_start=True, early_stopping=True)

            self.random_search_cv(nn, param_grid, X_train, y_train, X_test, y_test, "DefaultCreditCardMLP", cv=3)
        else:
            # SVM
            self.load_pretrained_models("svm_default_client_grid_model")

            # DTC
            self.load_pretrained_models("tree_default_client_grid_model")

            # RFC
            self.load_pretrained_models("random_forest_default_client_grid_model")

            # LR
            self.load_pretrained_models("logistic_default_client_random_model")

            # Adaboost
            self.load_pretrained_models("adaboost_default_client_grid_model")

            # KNN
            self.load_pretrained_models("knearest_default_client_grid_model")

            # GNB
            self.load_pretrained_models("gaussian_default_client_grid_model")

            # MLP
            self.load_pretrained_models("mlp_default_client_random_model")

    def Breast_Cancer_Wisconsin(self):
        print('Running classification for 3.Breast Cancer Wisconsin dataset')

        ''' DATASET WBC'''

        df_wbc = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
            delimiter=",", header=None,
            names=['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                   'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses',
                   'Class'])

        df_wbc = df_wbc.drop(columns="id")
        df_wbc = df_wbc.replace('?', 0).astype(int)

        mean_bn = df_wbc["Bare Nuclei"].mean()

        df_wbc.loc[df_wbc['Bare Nuclei'] == 0, ['Bare Nuclei']] = int(mean_bn)

        X = df_wbc.loc[:, 'Clump Thickness':'Mitoses']
        y = df_wbc.loc[:, 'Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        ''' KNN CLASSIFICATION'''

        print('Running KNN Classifier\n')
        param_grid = {
            "n_neighbors": np.arange(5, 50, 5),
            "weights": ['uniform', 'distance'],
            "leaf_size": np.arange(5, 100, 10)
        }

        self.grid_search_cv(self, sklearn.neighbors.KNeighborsClassifier(), param_grid, 5, X_train, y_train)

        ''' Decision Tree CLASSIFICATION'''

        print('Running Decision Tree Classifier\n')
        param_grid = {'max_depth': np.arange(5, 50),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.grid_search_cv(self, Tree.DecisionTreeClassifier(random_state=0), param_grid, 5, X_train, y_train)

        ''' SVM CLASSIFICATION'''

        print('Running SVM Classifier\n')
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': np.logspace(0, 3, 4),
            'gamma': np.logspace(-2, 1, 4)
        }

        self.random_search_cv(self, sklearn.svm.SVC(random_state=0), param_grid, 3, X_train, y_train)

        '''RANDOM FOREST CLASSIFIER'''

        print('Running Random Forest Classifier\n')
        param_grid = {'n_estimators': np.arange(5, 20, 5),
                      'max_depth': np.arange(5, 50),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.random_search_cv(self, sklearn.ensemble.RandomForestClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''ADABOOST CLASSIFIER'''

        print('Running Adaboost Classifier\n')
        param_grid = {'n_estimators': np.arange(25, 75, 5),
                      'learning_rate': np.arange(0.1, 1.1, 0.1),
                      'algorithm': ['SAMME', 'SAMME.R']
                      }

        self.random_search_cv(self, sklearn.ensemble.AdaBoostClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''LOGISTIC REGRESSION CLASSIFIER'''

        print('Running Logistic Regression Classifier\n')
        param_grid = {
            'C': np.logspace(0, 3, 4),
            'fit_intercept': [True, False],
            'max_iter': [50, 100, 150],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        self.random_search_cv(self, sklearn.linear_model.LogisticRegression(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

        print('Running Gaussian Naive Bayes Classifier\n')
        param_grid = {
            "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
        self.grid_search_cv(self, GaussianNB(), param_grid, 5, X_train, y_train)

        '''Neural Network Classifier'''

        mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                   learning_rate='adaptive', random_state=0, verbose=True,
                                                   warm_start=True, early_stopping=True)

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.1, 1.1, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
        }

        self.random_search_cv(self, mlp, param_grid, 5, X_train, y_train)
        '''DATASET WDBC'''
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                         delimiter=",", header=None,
                         names=['id', 'Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness',
                                'compactness', 'concavity', 'concave points', 'symmetry',
                                'fractal dimension', 'radius SE', 'texture SE', 'perimeter SE', 'area SE',
                                'smoothness SE', 'compactness SE', 'concavity SE', 'concave points SE',
                                'symmetry SE', 'fractal dimension SE',
                                'worst radius', 'worst texture', 'worst perimeter',
                                'worst area', 'worst smoothness', 'worst compactness',
                                'worst concavity', 'worst concave points', 'worst symmetry',
                                'worst fractal dimension'])

        df = df.drop(columns="id")

        X = df.loc[:, 'radius':'worst fractal dimension']
        y = df.loc[:, 'Diagnosis']
        y = y.replace({'B': 0, 'M': 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        ''' KNN CLASSIFICATION'''

        print('Running KNN Classifier\n')
        param_grid = {
            "n_neighbors": np.arange(5, 50, 5),
            "weights": ['uniform', 'distance'],
            "leaf_size": np.arange(5, 100, 10)
        }

        self.grid_search_cv(self, sklearn.neighbors.KNeighborsClassifier(), param_grid, 5, X_train, y_train)

        ''' Decision Tree CLASSIFICATION'''

        print('Running Decision Tree Classifier\n')
        param_grid = {'max_depth': np.arange(5, 50),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.grid_search_cv(self, Tree.DecisionTreeClassifier(random_state=0), param_grid, 5, X_train, y_train)

        ''' SVM CLASSIFICATION'''

        print('Running SVM Classifier\n')
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': np.logspace(0, 3, 4),
            'gamma': np.logspace(-2, 1, 4)
        }

        self.random_search_cv(self, sklearn.svm.SVC(random_state=0), param_grid, 3, X_train, y_train)

        '''RANDOM FOREST CLASSIFIER'''

        print('Running Random Forest Classifier\n')
        param_grid = {'n_estimators': np.arange(5, 20, 5),
                      'max_depth': np.arange(5, 50, 3),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.random_search_cv(self, sklearn.ensemble.RandomForestClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''ADABOOST CLASSIFIER'''

        print('Running Adaboost Classifier\n')
        param_grid = {'n_estimators': np.arange(25, 75, 5),
                      'learning_rate': np.arange(0.1, 1.1, 0.1),
                      'algorithm': ['SAMME', 'SAMME.R']
                      }

        self.random_search_cv(self, sklearn.ensemble.AdaBoostClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''LOGISTIC REGRESSION CLASSIFIER'''

        print('Running Logistic Regression Classifier\n')
        param_grid = {
            'C': np.logspace(0, 3, 4),
            'fit_intercept': [True, False],
            'max_iter': [50, 100, 150],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        self.random_search_cv(self, sklearn.linear_model.LogisticRegression(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

        print('Running Gaussian Naive Bayes Classifier\n')
        param_grid = {
            "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
        self.grid_search_cv(self, GaussianNB(), param_grid, 5, X_train, y_train)

        '''Neural Network Classifier'''

        mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                   learning_rate='adaptive', random_state=0, verbose=True,
                                                   warm_start=True, early_stopping=True)

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.1, 1.1, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
        }

        self.random_search_cv(self, mlp, param_grid, 5, X_train, y_train)
        '''DATASET WPBC'''

        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data",
                         delimiter=",", header=None, names=['id', 'Outcome', 'Time',
                                                            'radius', 'texture', 'perimeter', 'area', 'smoothness',
                                                            'compactness', 'concavity', 'concave points', 'symmetry',
                                                            'fractal dimension',
                                                            'radius SE', 'texture SE', 'perimeter SE', 'area SE',
                                                            'smoothness SE', 'compactness SE', 'concavity SE',
                                                            'concave points SE', 'symmetry SE', 'fractal dimension SE',
                                                            'worst radius', 'worst texture', 'worst perimeter',
                                                            'worst area', 'worst smoothness', 'worst compactness',
                                                            'worst concavity', 'worst concave points', 'worst symmetry',
                                                            'worst fractal dimension', 'Tumor size',
                                                            'Lymph node status'])

        df = df.drop(columns="id")

        df.replace('[\?,)]', '-0', regex=True, inplace=True)

        df_temp = df.replace('[\?,)]', '-0', regex=True)

        df_temp["Lymph node status"] = pd.DataFrame(df_temp["Lymph node status"]).astype(int)

        # export_csv = df.to_csv (r'drive/My Drive/data_replaced.csv', index = None, header=True)

        mean = df_temp["Lymph node status"].mean()

        df.loc[df["Lymph node status"] == '-0', ['Lymph node status']] = int(mean)

        X = df.loc[:, 'Time':'Lymph node status']
        y = df.loc[:, 'Outcome']
        y = y.replace({'N': 0, 'R': 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        ''' KNN CLASSIFICATION'''

        print('Running KNN Classifier\n')
        param_grid = {
            "n_neighbors": np.arange(5, 50, 5),
            "weights": ['uniform', 'distance'],
            "leaf_size": np.arange(5, 100, 10)
        }

        self.grid_search_cv(self, sklearn.neighbors.KNeighborsClassifier(), param_grid, 5, X_train, y_train)

        ''' Decision Tree CLASSIFICATION'''

        print('Running Decision Tree Classifier\n')
        param_grid = {
            'max_depth': np.arange(5, 50),
            'max_leaf_nodes': np.arange(5, 50, 5),
            'criterion': ['gini', 'entropy']
        }

        self.grid_search_cv(self, Tree.DecisionTreeClassifier(random_state=0), param_grid, 5, X_train, y_train)

        ''' SVM CLASSIFICATION'''

        print('Running SVM Classifier\n')
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': np.logspace(0, 3, 4),
            'gamma': np.logspace(-2, 1, 4)
        }

        self.random_search_cv(self, sklearn.svm.SVC(random_state=0), param_grid, 3, X_train, y_train)

        '''RANDOM FOREST CLASSIFIER'''

        print('Running Random Forest Classifier\n')
        param_grid = {'n_estimators': np.arange(5, 20, 5),
                      'max_depth': np.arange(5, 50),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.random_search_cv(self, sklearn.ensemble.RandomForestClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''ADABOOST CLASSIFIER'''

        print('Running Adaboost Classifier\n')
        param_grid = {'n_estimators': np.arange(25, 75, 5),
                      'learning_rate': np.arange(0.1, 1.1, 0.1),
                      'algorithm': ['SAMME', 'SAMME.R']
                      }

        self.random_search_cv(self, sklearn.ensemble.AdaBoostClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''LOGISTIC REGRESSION CLASSIFIER'''

        print('Running Logistic Regression Classifier\n')
        param_grid = {
            'C': np.logspace(0, 3, 4),
            'fit_intercept': [True, False],
            'max_iter': [50, 100, 150],
            'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
        }
        self.random_search_cv(self, sklearn.linear_model.LogisticRegression(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

        print('Running Gaussian Naive Bayes Classifier\n')
        param_grid = {
            "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
        self.grid_search_cv(self, GaussianNB(), param_grid, 5, X_train, y_train)

        '''Neural Network Classifier'''

        mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                   learning_rate='adaptive', random_state=0, verbose=True,
                                                   warm_start=True, early_stopping=True)

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.1, 1.1, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
        }

        self.random_search_cv(self, mlp, param_grid, 5, X_train, y_train)

    def Statlog_Australian(self):
        print('Running classification for 4.Statlog Australian dataset')

    def Statlog_German(self):
        print('Running classification for 5.Statlog German dataset')

        # %%
        '''
        ### **Preprocessing**
        '''

        # %%
        file = "../Datasets/5_GermanData.xlsx"
        df = pd.read_excel(file, header=None)
        # print(df)
        data = pd.DataFrame(df)
        data = data.values
        print(data)

        data = data.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:23], data[:, 24], test_size=0.20, random_state=0)

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # %%
        '''
        ### **Logistic Regression**
        '''

        # %%
        lr = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=10000)

        param = {'solver': ["sag", "saga", "liblinear"], 'C': [0.1, 0.2, 0.5, 1, 1.5, 2, 5, 7, 10, 12, 15]}

        self.random_search_cv(lr, param, X_train, y_train, X_test, y_test)
        # model = sklearn.model_selection.GridSearchCV(estimator=lr, param_grid=param, cv=5, scoring='roc_auc',
        #                                              verbose=5).fit(X_train, y_train)
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # print("Validation Score:", model.best_score_)
        # print("Testing Score:", model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'Logistic_Regression.sav'
        # pickle.dump(model, open(filename, 'wb'))
        # filename = 'Logistic_Regression_best_param.sav'
        # pickle.dump(model.best_params_, open(filename, 'wb'))

        # %%
        '''
        ### K-**Neighbors**
        '''

        # %%
        k_n = sklearn.neighbors.KNeighborsClassifier()

        param = {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20, 50, 100, 200, 500]}

        # model = sklearn.model_selection.GridSearchCV(estimator=k_n, param_grid=param, cv=5).fit(X_train, y_train)
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # print("Validation Score:", model.best_score_)
        # print("Testing Score:", model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'K_Neighbors.sav'
        # pickle.dump(model, open(filename, 'wb'))
        # filename = 'K_Neighbors_best_param.sav'
        # pickle.dump(model.best_params_, open(filename, 'wb'))

        # %%
        '''
        ### **SVM**
        '''

        # %%
        svm = sklearn.svm.SVC(random_state=0)

        param = dict(kernel=['rbf', 'linear'],
                     degree=[1, 2, 3],
                     C=Stats.reciprocal(0.01, 2),
                     gamma=Stats.reciprocal(0.01, 2))

        # model = sklearn.model_selection.RandomizedSearchCV(estimator=svm, param_distributions=param, verbose=10, cv=5,
        #                                                    random_state=0).fit(X_train, y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # print("Validation Score:", model.best_score_)
        # print("Testing Score:", model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'SVM.sav'
        # pickle.dump(model, open(filename, 'wb'))
        # filename = 'SVM_best_param.sav'
        # pickle.dump(model.best_params_, open(filename, 'wb'))

        # %%
        '''
        ### **Decision Tree**
        '''

        # %%
        dt = sklearn.tree.DecisionTreeClassifier(random_state=0)

        param = {'max_depth': np.arange(1, 20, 1),
                 'splitter': ['best', 'random'],
                 'max_features': np.arange(1, 19, 1),
                 'min_samples_split': np.arange(2, 20, 1)}

        # model = sklearn.model_selection.GridSearchCV(estimator=dt, param_grid=param, verbose=3, cv=5).fit(X_train,
        #                                                                                                   y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)

        # %%
        # model = sklearn.tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
        #                                             max_features=11, max_leaf_nodes=None,
        #                                             min_impurity_decrease=0.0, min_impurity_split=None,
        #                                             min_samples_leaf=1, min_samples_split=7,
        #                                             min_weight_fraction_leaf=0.0, presort=False,
        #                                             random_state=0, splitter='random').fit(X_train, y_train)
        #
        # # %%
        # print(model.score)
        # print(model.score(X_train, y_train))
        # print(model.score(X_test, y_test))
        #
        # # %%
        # filename = 'Decision_Tree.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Decision_Tree_best_param.sav'
        # pickle.dump(model.get_params, open(path + filename, 'wb'))
        # Best Params:  {'max_depth': 6, 'max_features': 11, 'min_samples_split': 7, 'splitter': 'random'}

        # %%
        '''
        ### **Random Forest**
        '''

        # %%
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0)

        param = {'max_depth': np.arange(1, 20, 1),
                 'max_features': np.array([5, 10, 15, 20]),
                 'min_samples_split': np.array([2, 3, 5])}

        # model = sklearn.model_selection.GridSearchCV(estimator=rf, param_grid=param, verbose=3, cv=5).fit(X_train,
        #                                                                                                   y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # print("Validation Score:", model.best_score_)
        # print("Testing Score:", model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'Random_Forest.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Random_Forest_best_param.sav'
        # pickle.dump(model.best_params_, open(path + filename, 'wb'))

        # %%

        # %%
        '''

        '''

        # %%
        '''
        ## **Ada Boost**
        '''

        # %%
        ada = sklearn.ensemble.AdaBoostClassifier(random_state=0)

        param = {'n_estimators': np.arange(50, 250, 10), 'algorithm': ['SAMME.R', 'SAMME']}

        # model = sklearn.model_selection.GridSearchCV(estimator=ada, param_grid=param, verbose=10, cv=5).fit(X_train,
        #                                                                                                     y_train)
        #
        # print('Best Score: ', model.best_score_)
        # print('Best Params: ', model.best_params_)
        #
        # # %%
        # print("Validation Score:", model.best_score_)
        # print("Testing Score:", model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'AdaBoostClassifier.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'AdaBoostClassifier_best_param.sav'
        # pickle.dump(model.best_params_, open(path + filename, 'wb'))

        # %%
        '''
        ## **Neural Network**
        '''

        # %%
        mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                   learning_rate='adaptive', random_state=0, verbose=True,
                                                   warm_start=True, early_stopping=True)

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": Stats.reciprocal(0.001, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
        }

        # model = sklearn.model_selection.RandomizedSearchCV(mlp, param_grid, n_iter=30, cv=5, scoring="accuracy",
        #                                                    verbose=5).fit(X_train, y_train)
        #
        # # %%
        # print(model.best_estimator_)
        # print(model.best_score_)
        # print(model.best_estimator_.score(X_test, y_test))
        #
        # # %%
        # filename = 'Neural_Network.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Neural_Network_best_param.sav'
        # pickle.dump(model.best_params_, open(path + filename, 'wb'))

        # %%
        '''
        ## **Guassian Naive Bayes Classification**
        '''

        # %%
        model = sklearn.naive_bayes.GaussianNB().fit(X_train, y_train)

        print(gb.score(X_train, y_train))
        print(gb.score(X_test, y_test))

        # %%
        # filename = 'Guassain_Naives_Bayes.sav'
        # pickle.dump(model, open(path + filename, 'wb'))
        # filename = 'Guassain_Naives_Bayes_best_param.sav'
        # pickle.dump(model.get_params, open(path + filename, 'wb'))

        # %%
        '''
        ## **Testing**
        '''

        # %%
        # model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(24, 2), random_state=0, max_iter=1000).fit(
        #     X_train, y_train)
        #
        # print(model.score(X_train, y_train))
        # print(model.score(X_test, y_test))

    def Steel_Plates_Faults(self, userResponse):
        print('Running classification for 6.Steel Plates Faults dataset')
        if userResponse is "2":
            df = pd.read_excel(DATASETS + "Faults.xlsx", header=0)
            df.columns = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter",
                          "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
                          "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
                          "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index",
                          "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
                          "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas", "Pastry", "Z_Scratch", "K_Scratch",
                          "Stains", "Dirtiness", "Bumps", "Other_Faults"]
            classes = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
            df['class'] = ''
            for i in range(df.shape[0]):
                if df.at[i, 'Pastry'] == 1:
                    df.at[i, 'class'] = 'Pastry'
                if df.at[i, 'Z_Scratch'] == 1:
                    df.at[i, 'class'] = 'Z_Scratch'
                if df.at[i, 'K_Scratch'] == 1:
                    df.at[i, 'class'] = 'K_Scratch'
                if df.at[i, 'Stains'] == 1:
                    df.at[i, 'class'] = 'Stains'
                if df.at[i, 'Dirtiness'] == 1:
                    df.at[i, 'class'] = 'Dirtiness'
                if df.at[i, 'Bumps'] == 1:
                    df.at[i, 'class'] = 'Bumps'
                if df.at[i, 'Other_Faults'] == 1:
                    df.at[i, 'class'] = 'Other_Faults'
            df.drop(["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"], axis=1,
                    inplace=True)

            df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
            X_train = df_train.iloc[:, :27]
            y_train = df_train.iloc[:, 27]
            X_test = df_test.iloc[:, :27]
            y_test = df_test.iloc[:, 27]
            # print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)

            scaler = Preprocessing.StandardScaler().fit \
                (X_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 22, 23]])
            encoder = Preprocessing.LabelEncoder()

            encoder.fit(classes)
            y_train_labels = encoder.transform(y_train)
            y_test_labels = encoder.transform(y_test)

            X_train_scaled = X_train.copy(deep=True)
            X_test_scaled = X_test.copy(deep=True)
            X_train_scaled.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 22, 23]] = scaler.transform(
                X_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 22, 23]])
            X_test_scaled.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 22, 23]] = scaler.transform(
                X_test.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 22, 23]])

            # SVM
            params_grid = {'gamma': [1e-3, 1e-4],
                           'C': [1, 10, 100, 1000]}
            svm = sklearn.svm.SVC(kernel="rbf", random_state=0)
            self.grid_search_cv(svm, params_grid, X_train_scaled, y_train_labels, X_test_scaled, y_test_labels,
                                "SteelFaultsSVM", cv=3)

            # DTC
            params_grid = {'max_depth': np.arange(4, 10),
                           'max_leaf_nodes': [5, 10, 20, 100],
                           'min_samples_split': [2, 5, 10, 20]
                           }
            dtc = Tree.DecisionTreeClassifier(random_state=0, criterion='gini')
            self.grid_search_cv(dtc, params_grid, X_train_scaled, y_train_labels, X_test_scaled, y_test_labels,
                                "SteelFaultsDTC", cv=3)

            # RFC
            params_grid = {'max_depth': np.arange(4, 10),
                           'max_leaf_nodes': [5, 10, 20],
                           'min_samples_split': [2, 5],
                           'n_estimators': [estimator for estimator in (2 ** i for i in range(0, 8))]}

            rfc = Ensemble.RandomForestClassifier(random_state=0, criterion='gini')
            self.grid_search_cv(rfc, params_grid, X_train_scaled, y_train_labels, X_test_scaled, y_test_labels,
                                "SteelFaultsRFC", cv=3)

            # LR
            params_grid = {
                "penalty": ['l2'],
                "C": Stats.reciprocal(0.001, 1000),
                "fit_intercept": [True, False],
                "solver": ['lbfgs', 'sag', 'saga'],
                "max_iter": [100, 200, 300, 400, 500]
            }
            lr = Linear.LogisticRegression(multi_class='auto', random_state=0)
            self.random_search_cv(lr, params_grid, X_train_scaled, y_train_labels, X_test_scaled, y_test_labels,
                                  "DefaultCreditCardLR", cv=3)

            # Adaboost
            params_grid = {
                "n_estimators": [50, 70, 100, 120, 150, 200, 250],
                "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
                "algorithm": ["SAMME", "SAMME.R"]
            }
            adaboost = Ensemble.AdaBoostClassifier(random_state=0)
            self.grid_search_cv(adaboost, params_grid, X_train, y_train_labels, X_test, y_test_labels,
                                "SteelFaultsAdaboost", cv=3)

            # KNN
            params_grid = {
                "n_neighbors": [10, 50, 100],
                "weights": ['uniform', 'distance'],
                "leaf_size": [15, 30, 50, 100]
            }
            knn = Neighbors.KNeighborsClassifier()
            self.grid_search_cv(knn, params_grid, X_train_scaled, y_train_labels, X_test_scaled, y_test_labels,
                                "SteelFaultsKNN", cv=3)

            # GNB
            params_grid = {
                "var_smoothing": [1e-07, 1e-08, 1e-09]
            }
            self.grid_search_cv(GaussianNB(), params_grid, X_train_scaled, y_train, X_test_scaled, y_test,
                                "SteelFaultsGaussian", cv=3)

            # MLP
            params_grid = {
                "solver": ['adam', 'sgd'],
                "learning_rate_init": Stats.reciprocal(0.001, 0.1),
                "hidden_layer_sizes": [(128, 64, 32), (512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
            }
            mlp = NN.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                   learning_rate='adaptive', verbose=True, warm_start=True,
                                   early_stopping=True)
            self.random_search_cv(mlp, params_grid, X_train_scaled, y_train_labels, X_test_scaled, y_test_labels,
                                  "SteelFaultsMLP", cv=3)
        else:
            # SVM
            self.load_pretrained_models("svm_faults_grid_model")

            # DTC
            self.load_pretrained_models("tree_faults_grid_model")

            # RFC
            self.load_pretrained_models("random_forest_faults_grid_model")

            # LR
            self.load_pretrained_models("logistic_faults_grid_model")

            # Adaboost
            self.load_pretrained_models("adaboost_faults_grid_model")

            # KNN
            self.load_pretrained_models("kNearest_faults_grid_model")

            # GNB
            self.load_pretrained_models("gaussian_faults_grid_model")

            # MLP
            self.load_pretrained_models("mlp_faults_grid_model")

    def Adult(self):
        print('Running classification for 7.Adult dataset')

        labelencoder_data = sklearn.preprocessing.LabelEncoder()

        scaler = sklearn.preprocessing.StandardScaler()

        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", delimiter=",",
                         header=None,
                         names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                'hours-per-week', 'native-country', 'income'])
        df_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                              delimiter=",", header=None,
                              names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                     'hours-per-week', 'native-country', 'income'], skiprows=1)

        #### TRAINING DATA
        df.replace('[\?,)]', 'N/A', regex=True, inplace=True)

        df['workclass'] = labelencoder_data.fit_transform(df['workclass'])
        df['marital-status'] = labelencoder_data.fit_transform(df['marital-status'])
        df['education'] = labelencoder_data.fit_transform(df['education'])
        df['occupation'] = labelencoder_data.fit_transform(df['occupation'])
        df['relationship'] = labelencoder_data.fit_transform(df['relationship'])
        df['race'] = labelencoder_data.fit_transform(df['race'])
        df['sex'] = labelencoder_data.fit_transform(df['sex'])
        df['native-country'] = labelencoder_data.fit_transform(df['native-country'])
        X_train = df.loc[:, 'age': 'native-country']

        y = df.loc[:, 'income']
        y = labelencoder_data.fit_transform(y)
        y_train = pd.DataFrame(y)

        ### TESTING DATA

        df_test.replace('[\?,)]', 'N/A', regex=True, inplace=True)

        df_test['workclass'] = labelencoder_data.fit_transform(df_test['workclass'])
        df_test['marital-status'] = labelencoder_data.fit_transform(df_test['marital-status'])

        df_test['education'] = labelencoder_data.fit_transform(df_test['education'])
        df_test['occupation'] = labelencoder_data.fit_transform(df_test['occupation'])
        df_test['relationship'] = labelencoder_data.fit_transform(df_test['relationship'])
        df_test['race'] = labelencoder_data.fit_transform(df_test['race'])
        df_test['sex'] = labelencoder_data.fit_transform(df_test['sex'])
        df_test['native-country'] = labelencoder_data.fit_transform(df_test['native-country'])
        X_test = df_test.loc[:, 'age': 'native-country']

        y_test = df_test.loc[:, 'income']
        y_test = labelencoder_data.fit_transform(y_test)
        y_test = pd.DataFrame(y_test)

        ''' KNN CLASSIFICATION'''

        print('Running KNN Classifier\n')
        param_grid = {
            "n_neighbors": np.arange(5, 40),
            "weights": ['uniform', 'distance']
        }

        self.grid_search_cv(self, sklearn.neighbors.KNeighborsClassifier(), param_grid, 5, X_train, y_train)

        ''' Decision Tree CLASSIFICATION'''

        print('Running Decision Tree Classifier\n')
        param_grid = {
            'max_depth': np.arange(5, 50, 5),
            'max_leaf_nodes': np.arange(5, 50, 5),
            'criterion': ['gini', 'entropy']
        }

        self.grid_search_cv(self, Tree.DecisionTreeClassifier(random_state=0), param_grid, X_train, y_train)

        ''' SVM CLASSIFICATION'''

        print('Running SVM Classifier\n')
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': np.logspace(0, 3, 2),
            'gamma': np.logspace(-2, 1, 2)
        }

        self.random_search_cv(self, sklearn.svm.SVC(random_state=0), param_grid, 2, X_train, y_train)

        '''RANDOM FOREST CLASSIFIER'''

        print('Running Random Forest Classifier\n')
        param_grid = {'n_estimators': np.arange(5, 20, 5),
                      'max_depth': np.arange(5, 50),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.random_search_cv(self, sklearn.ensemble.RandomForestClassifier(random_state=0), param_grid, 3, X_train,
                              y_train)

        '''ADABOOST CLASSIFIER'''

        print('Running Adaboost Classifier\n')
        param_grid = {'n_estimators': np.arange(25, 75, 5),
                      'learning_rate': np.arange(0.1, 1.1, 0.1),
                      'algorithm': ['SAMME', 'SAMME.R']
                      }

        self.random_search_cv(self, sklearn.ensemble.AdaBoostClassifier(random_state=0), param_grid, 5, X_train,
                              y_train)

        '''LOGISTIC REGRESSION CLASSIFIER'''

        print('Running Logistic Regression Classifier\n')
        param_grid = {
            'C': np.logspace(0, 3, 4),
            'fit_intercept': [True, False],
            'max_iter': [50, 100, 150],
            'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
        }
        self.random_search_cv(self, sklearn.linear_model.LogisticRegression(random_state=0), param_grid, 3, X_train,
                              y_train)

        '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

        print('Running Gaussian Naive Bayes Classifier\n')
        param_grid = {
            "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
        self.grid_search_cv(self, GaussianNB(), param_grid, 5, X_train, y_train)

        '''Neural Network Classifier'''

        mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                   learning_rate='adaptive', random_state=0, verbose=True,
                                                   warm_start=True, early_stopping=True)

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.1, 1.1, 0.1),
            "hidden_layer_sizes": [(128,), (128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
        }

        self.random_search_cv(self, mlp, param_grid, 5, X_train, y_train)

    def Yeast(self):
        print('Running classification for 8.Yeast dataset')

    def Thoracic_Surgery_Data(self):
        print('Running classification for 9.Thoracic Surgery Data dataset')
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff",
                         delimiter=",", header=None, skiprows=21)

        X = df.loc[:, :15]
        X = X.replace({'F': 0, 'T': 1})
        X = X.replace({'OC11': 0, 'OC12': 1, 'OC13': 2, 'OC14': 3})
        X = X.replace({'PRZ0': 0, 'PRZ1': 1, 'PRZ2': 2})
        X = X.replace({'DGN1': 0, 'DGN2': 1, 'DGN3': 2, 'DGN4': 3, 'DGN5': 4, 'DGN6': 5, 'DGN8': 6})
        y = df.loc[:, 16]
        y = y.replace({'F': 0, 'T': 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        ''' KNN CLASSIFICATION'''

        print('Running KNN Classifier\n')
        param_grid = {
            "n_neighbors": np.arange(5, 50),
            "weights": ['uniform', 'distance'],
            "leaf_size": np.arange(5, 100, 5)
        }

        self.grid_search_cv(self, sklearn.neighbors.KNeighborsClassifier(), param_grid, 3, X_train, y_train)

        ''' Decision Tree CLASSIFICATION'''

        print('Running Decision Tree Classifier\n')
        param_grid = {
            'max_depth': np.arange(5, 50, 5),
            'max_leaf_nodes': np.arange(5, 50, 3),
            'criterion': ['gini', 'entropy']
        }

        self.grid_search_cv(self, Tree.DecisionTreeClassifier(random_state=0), param_grid, 3, X_train, y_train)

        ''' SVM CLASSIFICATION'''

        print('Running SVM Classifier\n')
        param_grid = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'C': np.logspace(0, 3, 4),
            'gamma': np.logspace(-2, 1, 4)
        }

        self.random_search_cv(self, sklearn.svm.SVC(random_state=0), param_grid, 3, X_train, y_train)

        '''RANDOM FOREST CLASSIFIER'''

        print('Running Random Forest Classifier\n')
        param_grid = {'n_estimators': np.arange(5, 20, 3),
                      'max_depth': np.arange(5, 50, 3),
                      'max_leaf_nodes': np.arange(5, 50, 5),
                      'criterion': ['gini', 'entropy']
                      }

        self.grid_search_cv(self, sklearn.ensemble.RandomForestClassifier(random_state=0), param_grid, 3, X_train,
                            y_train)

        '''ADABOOST CLASSIFIER'''

        print('Running Adaboost Classifier\n')
        param_grid = {'n_estimators': np.arange(25, 75, 5),
                      'learning_rate': np.arange(0.1, 1.1, 0.1),
                      'algorithm': ['SAMME', 'SAMME.R']
                      }

        self.random_search_cv(self, sklearn.ensemble.AdaBoostClassifier(random_state=0), param_grid, 3, X_train,
                              y_train)

        '''LOGISTIC REGRESSION CLASSIFIER'''

        print('Running Logistic Regression Classifier\n')
        param_grid = {
            'C': np.logspace(0, 3, 4),
            'fit_intercept': [True, False],
            'max_iter': [50, 100, 150],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        self.random_search_cv(self, sklearn.linear_model.LogisticRegression(random_state=0), param_grid, 3, X_train,
                              y_train)

        '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

        print('Running Gaussian Naive Bayes Classifier\n')
        param_grid = {
            "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
        self.grid_search_cv(self, GaussianNB(), param_grid, 3, X_train, y_train)

        '''Neural Network Classifier'''

        mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                   learning_rate='adaptive', random_state=0, verbose=True,
                                                   warm_start=True, early_stopping=True)

        param_grid = {
            "solver": ['adam', 'sgd'],
            "learning_rate_init": np.arange(0.1, 1.1, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
        }

        self.random_search_cv(self, mlp, param_grid, 5, X_train, y_train)

    def Seismic_Bumps(self, userResponse):
        print('Running classification for 10.Seismic Bumps dataset')
        if userResponse is "2":
            df = pd.read_csv(DATASETS + "seismic-bumps.csv", header=None);
            df.columns = ["seismic", "seismoacoustic", "shift", "genergy", "gpuls", "gdenergy", "gdpuls", "ghazard",
                          "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
                          "energy", "maxenergy", "class"]

            df.drop(["nbumps6", "nbumps7", "nbumps89"], axis=1, inplace=True)

            encoder = Preprocessing.LabelEncoder()

            encoder.fit(df['seismic'])
            df['seismic'] = encoder.transform(df['seismic'])

            encoder.fit(df['seismoacoustic'])
            df['seismoacoustic'] = encoder.transform(df['seismoacoustic'])

            encoder.fit(df['shift'])
            df['shift'] = encoder.transform(df['shift'])

            encoder.fit(df['ghazard'])
            df['ghazard'] = encoder.transform(df['ghazard'])

            df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)
            X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
            X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
            scaler = Preprocessing.StandardScaler().fit(X_train.iloc[:, [3, 4, 5, 6, 13, 14]])
            X_train_scaled = X_train.copy(deep=True)
            X_test_scaled = X_test.copy(deep=True)
            X_train_scaled.iloc[:, [3, 4, 5, 6, 13, 14]] = scaler.transform(X_train.iloc[:, [3, 4, 5, 6, 13, 14]])
            X_test_scaled.iloc[:, [3, 4, 5, 6, 13, 14]] = scaler.transform(X_test.iloc[:, [3, 4, 5, 6, 13, 14]])

            smote = SMOTETomek(ratio='auto', random_state=0)
            X_smote, y_smote = smote.fit_sample(X_train_scaled, y_train)
            y_smote = pd.DataFrame(data=y_smote, dtype=np.int64)
            y_smote.columns = ['class']
            X_smote = pd.DataFrame(data=X_smote)
            X_smote.columns = ["seismic", "seismoacoustic", "shift", "genergy", "gpuls", "gdenergy", "gdpuls",
                               "ghazard", "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "energy", "maxenergy"]
            df_train_smote = pd.concat([X_smote, y_smote], axis=1)
            df_train_smote_shuffle = df_train_smote.sample(frac=1, axis=0, random_state=0).reset_index(drop=True)
            X_smote_train, y_smote_train = df_train_smote_shuffle.iloc[:, :-1], df_train_smote_shuffle.iloc[:, -1]

            df_train_smote_shuffle['class'].value_counts().plot(kind='bar', title='Count (class)');

            # SVM
            params_grid = {
                'gamma': [1e-2, 1e-1, 1, 10, 100],
                'C': [1, 10, 100]
            }
            svm = sklearn.svm.SVC(kernel="rbf", random_state=0)
            self.grid_search_cv(svm, params_grid, X_smote_train, y_smote_train, X_test_scaled, y_test,
                                "SeismicBumpsSVM", cv=3)

            # DTC
            params_grid = {'max_depth': np.arange(4, 10),
                           'max_leaf_nodes': [5, 10, 20, 100],
                           'min_samples_split': [2, 5, 10, 20]
                           }
            dtc = Tree.DecisionTreeClassifier(random_state=0, criterion='gini')
            self.grid_search_cv(dtc, params_grid, X_smote_train, y_smote_train, X_test_scaled, y_test,
                                "SeismicBumpsDTC", cv=3)

            # RFC
            params_grid = {'max_depth': np.arange(4, 10),
                           'max_leaf_nodes': [5, 10, 20],
                           'min_samples_split': [2, 5],
                           'n_estimators': [estimator for estimator in (2 ** i for i in range(0, 8))]}
            rfc = Ensemble.RandomForestClassifier(random_state=0, criterion='gini')
            self.grid_search_cv(rfc, params_grid, X_smote_train, y_smote_train, X_test_scaled, y_test,
                                "SeismicBumpsRFC", cv=3)

            # Adaboost
            params_grid = {
                "n_estimators": [50, 70, 100, 120, 150, 200, 250],
                "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
                "algorithm": ["SAMME", "SAMME.R"]
            }
            adaboost = Ensemble.AdaBoostClassifier(random_state=0)
            self.grid_search_cv(adaboost, params_grid, X_smote_train, y_smote_train, X_test_scaled, y_test,
                                "SeismicBumpsAdaboost", cv=3)

            # KNN
            params_grid = {
                "n_neighbors": [10, 50, 100],
                "weights": ['uniform', 'distance'],
                "leaf_size": [15, 30, 50, 100]
            }
            self.grid_search_cv(Neighbors.KNeighborsClassifier(), params_grid, X_smote_train, y_smote_train,
                                X_test_scaled, y_test,
                                "SeismicBumpsKnn", cv=3)

            # LR
            params_grid = {
                "penalty": ['l2'],
                "C": Stats.reciprocal(0.001, 1000),
                "fit_intercept": [True, False],
                "solver": ['lbfgs', 'sag', 'saga'],
                "max_iter": [100, 200, 300, 400, 500]
            }
            self.random_search_cv(Linear.LogisticRegression(random_state=0), params_grid, X_smote_train, y_smote_train,
                                  X_test_scaled, y_test, "SeismicBumpsLR", cv=3)

            # GNB
            params_grid = {
                "var_smoothing": [1e-07, 1e-08, 1e-09]
            }
            self.grid_search_cv(GaussianNB(), params_grid, X_smote_train, y_smote_train,
                                X_test_scaled, y_test, "SeismicBumpsGaussian", cv=3)

            # MLP
            params_grid = {
                "solver": ['adam', 'sgd'],
                "learning_rate_init": Stats.reciprocal(0.001, 0.1),
                "hidden_layer_sizes": [(128, 64, 32), (512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
            }
            mlp = NN.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                   learning_rate='adaptive', verbose=True, warm_start=True, early_stopping=True)
            self.random_search_cv(mlp, params_grid, X_smote_train, y_smote_train, X_test_scaled, y_test,
                                  "SeismicBumpsMLP", cv=3)
        else:
            # SVM
            self.load_pretrained_models("svm_bumps_grid_model")

            # DTC
            self.load_pretrained_models("tree_bumps_grid_model")

            # RFC
            self.load_pretrained_models("random_bumps_grid_model")

            # LR
            self.load_pretrained_models("logistic_bumps_grid_model")

            # Adaboost
            self.load_pretrained_models("adaboost_bumps_grid_model")

            # KNN
            self.load_pretrained_models("kNearest_bumps_grid_model")

            # GNB
            self.load_pretrained_models("gaussian_bumps_grid_model")

            # MLP
            self.load_pretrained_models("mlp_bumps_grid_model")
