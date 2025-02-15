import itertools
import pickle
from src import plot
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
# from imblearn.combine import SMOTETomek
# from imblearn.over_sampling import SMOTE
from scipy.io import arff
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

    def load_pretrained_models(self, name, X_train, y_train, X_test, y_test):
        print("Loading PreTrained model: ", name)
        model = pickle.load(open(PRETRAINED_MODEL + name + ".sav", 'rb'))
        print("Testing Accuracy: ", model.score(X_test, y_test))
        print("Training Accuracy: ", model.score(X_train, y_train))

    def run_classifier(self,userResponse):
        print('Running classifiers for the following datasets: \n')
        self.Diabetic_Retinopathy(userResponse)
        # self.Default_of_credit_card_clients(userResponse)
        self.Breast_Cancer_Wisconsin(userResponse)
        # self.Statlog_Australian()
        self.Statlog_German(userResponse)
        # self.Steel_Plates_Faults(userResponse)
        # self.Adult(userResponse)
        # self.Yeast(userResponse)
        self.Thoracic_Surgery_Data(userResponse)
        # self.Seismic_Bumps(userResponse)

    def Diabetic_Retinopathy(self,userResponse):
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

        if(userResponse == '2'):

            '''Logistic Regression'''

            lr = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=10000)

            param = {'solver': ["sag", "saga", "liblinear"],
                     'C': [0.1, 0.2, 0.5, 1, 1.5, 2, 5, 7, 10, 12, 15]
                     }

            self.random_search_cv(lr, param, X_train, y_train, X_test, y_test, "Diabetic_LogisticRegression")

            '''
            ### K-**Neighbors**
            '''


            k_n = sklearn.neighbors.KNeighborsClassifier()

            param = {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20, 50, 100, 200, 500]}

            self.random_search_cv(k_n, param, X_train, y_train, X_test, y_test, "Diabetic_K_Neighbors")

            '''
            ### **SVM**
            '''

            svm = sklearn.svm.SVC(random_state=0)

            param = dict(kernel=['rbf', 'linear'],
                         degree=[1, 2, 3],
                         C=Stats.reciprocal(0.01, 2),
                         gamma=Stats.reciprocal(0.01, 2))

            self.random_search_cv(svm, param, X_train, y_train, X_test, y_test, "Diabetic_SVM")

            '''
            ### **Decision Tree**
            '''

            dt = sklearn.tree.DecisionTreeClassifier(random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'splitter': ['best', 'random'],
                     'max_features': np.arange(1, 19, 1),
                     'min_samples_split': np.arange(2, 20, 1)}

            self.grid_search_cv(dt, param, X_train, y_train, X_test, y_test, "Diabetic_Decision_Tree")

            '''
            ### **Random Forest**
            '''
            rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'max_features': np.arange(1, 19, 1),
                     'min_samples_split': np.arange(2, 20, 1)}

            self.random_search_cv(rf, param, X_train, y_train, X_test, y_test, "Diabetic_Random_Forest")

            '''
            ### **Gaussian naive Bayes classification**
            '''

            gb = sklearn.naive_bayes.GaussianNB().fit(X_train, y_train)
            name="Diabetic_Gaussian_Naive_Bayes"
            print("Testing Accuracy: ", gb.score(X_test, y_test))
            print("Training Accuracy: ", gb.score(X_train, y_train))
            pickle.dump(gb, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(gb.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(gb, name + " Learning Curve", X_train, y_train, (0.5, 1.01),cv=5)

            '''
            ### **Neural Network**
            '''

            nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(20, 15, 20, 2), random_state=0,
                                                         max_iter=1000).fit(X_train, y_train)
            name="Diabetic_Neural_Network"
            print("Testing Accuracy: ", nn.score(X_test, y_test))
            print("Training Accuracy: ", nn.score(X_train, y_train))
            pickle.dump(nn, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(nn.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(nn, name + " Learning Curve", X_train, y_train, (0.5, 1.01),cv=5)
            '''
            ### **Ada Boost**
            '''

            ada = sklearn.ensemble.AdaBoostClassifier(random_state=0)

            param = dict(n_estimators=np.arange(50, 250, 10),
                         algorithm=['SAMME.R', 'SAMME']
                         )

            self.grid_search_cv(ada, param, X_train, y_train, X_test, y_test, "Diabetic_Ada_Boost")
        else:

            self.load_pretrained_models("Diabetic_SVMModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_LogisticRegressionModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_Decision_TreeModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_Random_ForestModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_Ada_BoostModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_Neural_NetworkModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_Gaussian_Naive_BayesModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("Diabetic_K_NeighborsModel", X_train, y_train, X_test, y_test)

    def Default_of_credit_card_clients(self, userResponse):
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

        if userResponse is "2":
            print('Running classification for 2.Default of credit card clients dataset')
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
            self.load_pretrained_models("svm_default_client_grid_model", X_train, y_train, X_test, y_test, 3)

            # DTC
            self.load_pretrained_models("tree_default_client_grid_model", X_train, y_train, X_test, y_test, 3)

            # RFC
            self.load_pretrained_models("random_forest_default_client_grid_model", X_train, y_train, X_test, y_test, 3)

            # LR
            self.load_pretrained_models("logistic_default_client_random_model", X_train, y_train, X_test, y_test, 3)

            # Adaboost
            self.load_pretrained_models("adaboost_default_client_grid_model", X_train, y_train, X_test, y_test, 3)

            # KNN
            self.load_pretrained_models("knearest_default_client_grid_model", X_train, y_train, X_test, y_test, 3)

            # GNB
            self.load_pretrained_models("gaussian_default_client_grid_model", X_train, y_train, X_test, y_test, 3)

            # MLP
            self.load_pretrained_models("mlp_default_client_random_model", X_train, y_train, X_test, y_test, 3)

    def Breast_Cancer_Wisconsin(self, userResponse):
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

        if userResponse is "2":
            ''' KNN CLASSIFICATION'''

            print('Running KNN Classifier\n')
            param_grid = {
                "n_neighbors": np.arange(5, 50, 5),
                "weights": ['uniform', 'distance'],
                "leaf_size": np.arange(5, 100, 10)
            }
            knn = Neighbors.KNeighborsClassifier()
            self.grid_search_cv(knn, param_grid, X_train, y_train, X_test,
                                y_test, "kNN_WBC_model", 5)

            ''' Decision Tree CLASSIFICATION'''

            print('Running Decision Tree Classifier\n')
            param_grid = {'max_depth': np.arange(5, 50),
                          'max_leaf_nodes': np.arange(5, 50, 5),
                          'criterion': ['gini', 'entropy']
                          }

            dtc = Tree.DecisionTreeClassifier(random_state=0)
            self.grid_search_cv(dtc, param_grid, X_train, y_train, X_test,
                                y_test, "DecisionTree_WBC_model", 5)

            ''' SVM CLASSIFICATION'''

            print('Running SVM Classifier\n')
            param_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': np.logspace(0, 3, 4),
                'gamma': np.logspace(-2, 1, 4)
            }

            svm = sklearn.svm.SVC(random_state=0)
            self.random_search_cv(svm, param_grid, X_train, y_train, X_test, y_test,
                                  "Svm_WBC_model", 5)

            '''RANDOM FOREST CLASSIFIER'''

            print('Running Random Forest Classifier\n')
            param_grid = {'n_estimators': np.arange(5, 20, 5),
                          'max_depth': np.arange(5, 50),
                          'max_leaf_nodes': np.arange(5, 50, 5),
                          'criterion': ['gini', 'entropy']
                          }

            rfc = Ensemble.RandomForestClassifier(random_state=0)
            self.random_search_cv(rfc, param_grid, X_train,
                                  y_train, X_test, y_test, "RandomForest_WBC_model", 5)

            '''ADABOOST CLASSIFIER'''

            print('Running Adaboost Classifier\n')
            param_grid = {'n_estimators': np.arange(25, 75, 5),
                          'learning_rate': np.arange(0.1, 1.1, 0.1),
                          'algorithm': ['SAMME', 'SAMME.R']
                          }

            adaboost = Ensemble.AdaBoostClassifier(random_state=0)
            self.random_search_cv(adaboost, param_grid, X_train,
                                  y_train, X_test, y_test, "Adaboost_WBC_model", 5)

            '''LOGISTIC REGRESSION CLASSIFIER'''

            print('Running Logistic Regression Classifier\n')
            param_grid = {
                'C': np.logspace(0, 3, 4),
                'fit_intercept': [True, False],
                'max_iter': [50, 100, 150],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }

            lr = Linear.LogisticRegression(multi_class='auto', random_state=0)
            self.random_search_cv(lr, param_grid, X_train,
                                  y_train, X_test, y_test, "Logistic_WBC_model", 5)

            '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

            print('Running Gaussian Naive Bayes Classifier\n')
            param_grid = {
                "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test, "Gaussian_WBC_model",
                                5)

            '''Neural Network Classifier'''

            mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                       learning_rate='adaptive', random_state=0, verbose=True,
                                                       warm_start=True, early_stopping=True)

            param_grid = {
                "solver": ['adam'],
                "learning_rate_init": np.arange(0.1, 1.1, 0.1),
                "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
            }

            self.random_search_cv(mlp, param_grid, X_train, y_train, X_test, y_test, "Mlp_WBC_model", 5)
        else:
            # SVM
            self.load_pretrained_models("Svm_WBC_modelModel", X_train, y_train, X_test, y_test)

            # DTC
            self.load_pretrained_models("DecisionTree_WBC_modelModel", X_train, y_train, X_test, y_test)

            # RFC
            self.load_pretrained_models("RandomForest_WBC_modelModel", X_train, y_train, X_test, y_test)

            # LR
            self.load_pretrained_models("Logistic_WBC_modelModel", X_train, y_train, X_test, y_test)

            # Adaboost
            self.load_pretrained_models("Adaboost_WBC_modelModel", X_train, y_train, X_test, y_test)

            # KNN
            self.load_pretrained_models("kNN_WBC_modelModel", X_train, y_train, X_test, y_test)

            # GNB
            self.load_pretrained_models("Gaussian_WBC_modelModel", X_train, y_train, X_test, y_test)

            # MLP
            self.load_pretrained_models("Mlp_WBC_modelModel", X_train, y_train, X_test, y_test)

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

        if userResponse is "2":

            ''' KNN CLASSIFICATION'''

            print('Running KNN Classifier\n')
            param_grid = {
                "n_neighbors": np.arange(5, 50, 5),
                "weights": ['uniform', 'distance'],
                "leaf_size": np.arange(5, 100, 10)
            }
            knn = Neighbors.KNeighborsClassifier()
            self.grid_search_cv(knn, param_grid, X_train, y_train, X_test,
                                y_test, "knearest_WDBC_model")

            ''' Decision Tree CLASSIFICATION'''

            print('Running Decision Tree Classifier\n')
            param_grid = {'max_depth': np.arange(5, 50),
                          'max_leaf_nodes': np.arange(5, 50, 5),
                          'criterion': ['gini', 'entropy']
                          }

            dtc = Tree.DecisionTreeClassifier(random_state=0)

            self.grid_search_cv(dtc, param_grid, X_train, y_train, X_test,
                                y_test, "tree_WDBC_model")

            ''' SVM CLASSIFICATION'''

            print('Running SVM Classifier\n')
            param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': np.logspace(0, 3, 4),
                'gamma': np.logspace(-2, 1, 4)
            }

            svm = sklearn.svm.SVC(random_state=0)
            self.random_search_cv(svm, param_grid, X_train, y_train, X_test, y_test,
                                  "svm_WDBC_model")

            '''RANDOM FOREST CLASSIFIER'''

            print('Running Random Forest Classifier\n')
            param_grid = {'n_estimators': np.arange(5, 20, 5),
                          'max_depth': np.arange(5, 50, 3),
                          'max_leaf_nodes': np.arange(5, 50, 5),
                          'criterion': ['gini', 'entropy']
                          }

            rfc = Ensemble.RandomForestClassifier(random_state=0)

            self.random_search_cv(rfc, param_grid, X_train,
                                  y_train, X_test, y_test, "random_forest_WDBC_model")

            '''ADABOOST CLASSIFIER'''

            print('Running Adaboost Classifier\n')
            param_grid = {'n_estimators': np.arange(25, 75, 5),
                          'learning_rate': np.arange(0.1, 1.1, 0.1),
                          'algorithm': ['SAMME', 'SAMME.R']
                          }

            adaboost = Ensemble.AdaBoostClassifier(random_state=0)

            self.random_search_cv(adaboost, param_grid, X_train,
                                  y_train, X_test, y_test, "adaboost_WDBC_model")

            '''LOGISTIC REGRESSION CLASSIFIER'''

            print('Running Logistic Regression Classifier\n')
            param_grid = {
                'C': np.logspace(0, 3, 4),
                'fit_intercept': [True, False],
                'max_iter': [50, 100, 150],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }

            lr = Linear.LogisticRegression(multi_class='auto', random_state=0)

            self.random_search_cv(lr, param_grid, X_train,
                                  y_train, X_test, y_test, "logistic_WDBC_model")

            '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

            print('Running Gaussian Naive Bayes Classifier\n')
            param_grid = {
                "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test, "gaussian_WDBC_model")

            '''Neural Network Classifier'''

            mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                       learning_rate='adaptive', random_state=0, verbose=True,
                                                       warm_start=True, early_stopping=True)

            param_grid = {
                "solver": ['adam'],
                "learning_rate_init": np.arange(0.1, 1.1, 0.1),
                "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
            }

            self.random_search_cv(mlp, param_grid, X_train, y_train, X_test, y_test, "mlp_WDBC_model")
        else:
            # SVM
            self.load_pretrained_models("svm_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # DTC
            self.load_pretrained_models("tree_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # RFC
            self.load_pretrained_models("random_forest_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # LR
            self.load_pretrained_models("logistic_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # Adaboost
            self.load_pretrained_models("adaboost_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # KNN
            self.load_pretrained_models("knearest_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # GNB
            self.load_pretrained_models("gaussian_WDBC_modelModel", X_train, y_train, X_test, y_test)

            # MLP
            self.load_pretrained_models("mlp_WDBC_modelModel", X_train, y_train, X_test, y_test)

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

        if userResponse is "2":

            ''' KNN CLASSIFICATION'''

            print('Running KNN Classifier\n')
            param_grid = {
                "n_neighbors": np.arange(5, 50, 5),
                "weights": ['uniform', 'distance'],
                "leaf_size": np.arange(5, 100, 10)
            }
            knn = Neighbors.KNeighborsClassifier()
            self.grid_search_cv(knn, param_grid, X_train, y_train, X_test,
                                y_test, "knearest_WPBC_model")

            ''' Decision Tree CLASSIFICATION'''

            print('Running Decision Tree Classifier\n')
            param_grid = {
                'max_depth': np.arange(5, 50),
                'max_leaf_nodes': np.arange(5, 50, 5),
                'criterion': ['gini', 'entropy']
            }
            dtc = Tree.DecisionTreeClassifier(random_state=0)

            self.grid_search_cv(dtc, param_grid, X_train, y_train, X_test,
                                y_test, "tree_WPBC_model")

            ''' SVM CLASSIFICATION'''

            print('Running SVM Classifier\n')
            param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': np.logspace(0, 3, 4),
                'gamma': np.logspace(-2, 1, 4)
            }
            svm = sklearn.svm.SVC(random_state=0)

            self.random_search_cv(svm, param_grid, X_train, y_train, X_test, y_test,
                                  "svm_WPBC_model")

            '''RANDOM FOREST CLASSIFIER'''

            print('Running Random Forest Classifier\n')
            param_grid = {'n_estimators': np.arange(5, 20, 5),
                          'max_depth': np.arange(5, 50),
                          'max_leaf_nodes': np.arange(5, 50, 5),
                          'criterion': ['gini', 'entropy']
                          }
            rfc = Ensemble.RandomForestClassifier(random_state=0)

            self.random_search_cv(rfc, param_grid, X_train,
                                  y_train, X_test, y_test, "random_forest_WPBC_model")

            '''ADABOOST CLASSIFIER'''

            print('Running Adaboost Classifier\n')
            param_grid = {'n_estimators': np.arange(25, 75, 5),
                          'learning_rate': np.arange(0.1, 1.1, 0.1),
                          'algorithm': ['SAMME', 'SAMME.R']
                          }

            adaboost = Ensemble.AdaBoostClassifier(random_state=0)

            self.random_search_cv(adaboost, param_grid, X_train,
                                  y_train, X_test, y_test, "adaboost_WPBC_model")

            '''LOGISTIC REGRESSION CLASSIFIER'''

            print('Running Logistic Regression Classifier\n')
            param_grid = {
                'C': np.logspace(0, 3, 4),
                'fit_intercept': [True, False],
                'max_iter': [50, 100, 150],
                'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
            }

            lr = Linear.LogisticRegression(multi_class='auto', random_state=0)

            self.random_search_cv(lr, param_grid, X_train,
                                  y_train, X_test, y_test, "logistic_WPBC_model")

            '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

            print('Running Gaussian Naive Bayes Classifier\n')
            param_grid = {
                "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test, "gaussian_WPBC_model")

            '''Neural Network Classifier'''

            mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                       learning_rate='adaptive', random_state=0, verbose=True,
                                                       warm_start=True, early_stopping=True)

            param_grid = {
                "solver": ['adam'],
                "learning_rate_init": np.arange(0.1, 1.1, 0.1),
                "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
            }

            self.random_search_cv(mlp, param_grid, X_train, y_train, X_test, y_test, "mlp_WPBC_model")
        else:
            # SVM
            self.load_pretrained_models("svm_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # DTC
            self.load_pretrained_models("tree_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # RFC
            self.load_pretrained_models("random_forest_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # LR
            self.load_pretrained_models("logistic_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # Adaboost
            self.load_pretrained_models("adaboost_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # KNN
            self.load_pretrained_models("knearest_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # GNB
            self.load_pretrained_models("gaussian_WPBC_modelModel", X_train, y_train, X_test, y_test)

            # MLP
            self.load_pretrained_models("mlp_WPBC_modelModel", X_train, y_train, X_test, y_test)

    def Statlog_Australian(self, userResponse):
        print('Running classification for 4.Statlog Australian dataset')

        df = pd.read_excel('../Datasets/australian.xlsx', header=None,
                           index=False)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        X_train = df_train.iloc[:, 0:14]
        y_train = df_train.iloc[:, 14]

        X_test = df_test.iloc[:, 0:14]
        y_test = df_test.iloc[:, 14]

        scaler = Preprocessing.StandardScaler().fit(X_train.iloc[:, [12, 13]])
        X_train.loc[:, [12, 13]] = scaler.transform(X_train.iloc[:, [12, 13]])
        X_test.loc[:, [12, 13]] = scaler.transform(X_test.iloc[:, [12, 13]])

        if userResponse is "2":

            # SVM
            param_grid = {'C': [0.01, 0.1, 1],
                          'gamma': [0.01, 0.1, 1]}
            self.grid_search_cv(sklearn.svm.SVC(kernel='rbf', random_state=0), param_grid, X_train, y_train, X_test,
                                y_test, "StatlogAustralianSVM", 5)

            # DECISION TREE CLASSIFIER
            param_grid = {'max_depth': np.arange(3, 10),
                          'criterion': ['gini'],
                          'max_leaf_nodes': [5, 10, 20, 100],
                          'min_samples_split': [2, 5, 10, 20]}
            self.grid_search_cv(Tree.DecisionTreeClassifier(random_state=0), param_grid, X_train, y_train,
                                X_test, y_test, "StatlogAustralianDCT", 5)

            # RANDOM FOREST CLASSIFIER
            param_grid = {'max_depth': np.arange(3, 10),
                          'criterion': ['gini'],
                          'max_leaf_nodes': [5, 10],
                          'min_samples_split': [2, 5],
                          'n_estimators': [estimator for estimator in (2 ** i for i in range(0, 8))]}
            self.grid_search_cv(Ensemble.RandomForestClassifier(random_state=0), param_grid, X_train, y_train,
                                X_test, y_test, "StatlogAustralianRFC", cv=5)

            # ADABOOST CLASSIFIER
            param_grid = {
                "n_estimators": [30, 50, 70, 100],
                "learning_rate": [0.5, 0.7, 1, 2],
                "algorithm": ["SAMME", "SAMME.R"]
            }
            self.grid_search_cv(Ensemble.AdaBoostClassifier(random_state=0), param_grid,
                                X_train, y_train, X_test, y_test, "StatlogAustralianABC", 3)

            # LOGISTIC REGRESSION CLASSIFIER
            param_grid = {
                'penalty': ['l1', 'l2'],
                'C': Stats.reciprocal(0.001, 1000),
                'solver': ['liblinear']
            }
            self.random_search_cv(Linear.LogisticRegression(random_state=0), param_grid, X_train, y_train, X_test,
                                  y_test, "StatlogAustralianLRC", 3)

            # K NEAREST NAIGHBOUR CLASSIFIER

            param_grid = {
                "n_neighbors": [5, 10, 50],
                "weights": ['uniform', 'distance'],
                "leaf_size": [15, 30, 50, 100]
            }
            self.grid_search_cv(Neighbors.KNeighborsClassifier(), param_grid, X_train, y_train,
                                X_test, y_test, "StatlogAustralianKNN", 3)

            # GAUSSIAN NAIVE BAY

            param_grid = {
                "var_smoothing": [1e-07, 1e-08, 1e-09]
            }
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test, "StatlogAustralianGNB", 5)

            # NEURAL NETWORKS

            param_grid = {
                "solver": ['adam', 'sgd'],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "hidden_layer_sizes": [(128, 64, 32, 16, 2), (512, 256, 128, 64, 32)]
            }
            self.grid_search_cv(
                NN.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9, \
                                 learning_rate='adaptive', verbose=True, warm_start=True, \
                                 early_stopping=True), param_grid, X_train, y_train, X_test, y_test,
                "StatlogAustralianMLP", 3)

        else:
            # ADABOOST
            self.load_pretrained_models("StatlogAustralianABCModel", X_train, y_train, X_test, y_test)

            # DTC
            self.load_pretrained_models("StatlogAustralianDCTModel", X_train, y_train, X_test, y_test)

            # GNB
            self.load_pretrained_models("StatlogAustralianGNBModel", X_train, y_train, X_test, y_test)

            # KNN
            self.load_pretrained_models("StatlogAustralianKNNModel", X_train, y_train, X_test, y_test)

            # LR
            self.load_pretrained_models("StatlogAustralianLRCModel", X_train, y_train, X_test, y_test)

            # MLP
            self.load_pretrained_models("StatlogAustralianMLPModel", X_train, y_train, X_test, y_test)

            # RF
            self.load_pretrained_models("StatlogAustralianRFCModel", X_train, y_train, X_test, y_test)

            # SVM
            self.load_pretrained_models("StatlogAustralianSVMModel", X_train, y_train, X_test, y_test)

    def Statlog_German(self,userResponse):
        print('Running classification for 5.Statlog German dataset')
        '''
        ### **Preprocessing**
        '''

        file = "../Datasets/5_GermanData.xlsx"
        df = pd.read_excel(file, header=None)
        data = pd.DataFrame(df)
        data = data.values
        data = data.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:23], data[:, 24], test_size=0.20, random_state=0)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        if(userResponse=='2'):
            '''
            ### **Logistic Regression**
            '''

            lr = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=10000)

            param = {'solver': ["sag", "saga", "liblinear"], 'C': [0.1, 0.2, 0.5, 1, 1.5, 2, 5, 7, 10, 12, 15]}

            self.grid_search_cv(lr, param, X_train, y_train, X_test, y_test,"StatLogGerman_Logistic_Regression")

            '''
            ### K-**Neighbors**
            '''
            k_n = sklearn.neighbors.KNeighborsClassifier()

            param = {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20, 50, 100, 200, 500]}

            self.grid_search_cv(k_n, param, X_train, y_train, X_test, y_test, "StatLOgGerman_K_Neigbors")
            '''
            ### **SVM**
            '''
            svm = sklearn.svm.SVC(random_state=0)

            param = dict(kernel=['rbf', 'linear'],
                         degree=[1, 2, 3],
                         C=Stats.reciprocal(0.01, 2),
                         gamma=Stats.reciprocal(0.01, 2))

            self.random_search_cv(svm, param, X_train, y_train, X_test, y_test, "StatLOgGerman_SVM")
            '''
            ### **Decision Tree**
            '''
            dt = sklearn.tree.DecisionTreeClassifier(random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'splitter': ['best', 'random'],
                     'max_features': np.arange(1, 19, 1),
                     'min_samples_split': np.arange(2, 20, 1)}

            self.grid_search_cv(dt, param, X_train, y_train, X_test, y_test, "StatLOgGerman_Decision_Tree")
            '''
            ### **Random Forest**
            '''

            rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0)

            param = {'max_depth': np.arange(1, 20, 1),
                     'max_features': np.array([5, 10, 15, 20]),
                     'min_samples_split': np.array([2, 3, 5])}

            self.grid_search_cv(rf, param, X_train, y_train, X_test, y_test, "StatLOgGerman_Random_Forest")
            '''
            ## **Ada Boost**
            '''

            ada = sklearn.ensemble.AdaBoostClassifier(random_state=0)

            param = {'n_estimators': np.arange(50, 250, 10), 'algorithm': ['SAMME.R', 'SAMME']}

            self.grid_search_cv(ada, param, X_train, y_train, X_test, y_test, "StatLOgGerman_AdaBoost")

            '''
            ## **Neural Network**
            '''

            mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                       learning_rate='adaptive', random_state=0, verbose=True,
                                                       warm_start=True, early_stopping=True)

            param = {
                "solver": ['adam', 'sgd'],
                "learning_rate_init": Stats.reciprocal(0.001, 0.1),
                "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
            }

            self.random_search_cv(mlp, param, X_train, y_train, X_test, y_test, "StatLOgGerman_Neural_Network")
            '''
            ## **Guassian Naive Bayes Classification**
            '''
            gb = sklearn.naive_bayes.GaussianNB().fit(X_train, y_train)

            name = "StatLOgGerman_Gaussian_Naive_Bayes"
            print("Testing Accuracy: ", gb.score(X_test, y_test))
            print("Training Accuracy: ", gb.score(X_train, y_train))
            pickle.dump(gb, open(RESULTS_FOR_DEMO + "%sModel.sav" % name, 'wb'))
            pickle.dump(gb.get_params, open(RESULTS_FOR_DEMO + "%sBestParams.sav" % name, 'wb'))
            plot.plot_learning_curve(gb, name + " Learning Curve", X_train, y_train, (0.5, 1.01), cv=5)


        else:

            self.load_pretrained_models("StatlOgGerman_SVMModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_Logistic_RegressionModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_Decision_TreeModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_Random_ForestModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_AdaBoostModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_Neural_NetworkModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_Gaussian_Naive_BayesModel", X_train, y_train, X_test, y_test)
            self.load_pretrained_models("StatlOgGerman_K_NeigborsModel", X_train, y_train, X_test, y_test)

    def Steel_Plates_Faults(self, userResponse):
        print('Running classification for 6.Steel Plates Faults dataset')
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
        if userResponse is "2":
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
            self.load_pretrained_models("svm_faults_grid_model", X_train_scaled, y_train_labels, X_test_scaled,
                                        y_test_labels, 3)

            # DTC
            self.load_pretrained_models("tree_faults_grid_model", X_train_scaled, y_train_labels, X_test_scaled,
                                        y_test_labels, 3)

            # RFC
            self.load_pretrained_models("random_forest_faults_grid_model", X_train_scaled, y_train_labels,
                                        X_test_scaled,
                                        y_test_labels, 3)

            # LR
            self.load_pretrained_models("logistic_faults_grid_model", X_train_scaled, y_train_labels, X_test_scaled,
                                        y_test_labels, 3)

            # Adaboost
            self.load_pretrained_models("adaboost_faults_grid_model", X_train_scaled, y_train_labels, X_test_scaled,
                                        y_test_labels, 3)

            # KNN
            self.load_pretrained_models("kNearest_faults_grid_model", X_train_scaled, y_train_labels, X_test_scaled,
                                        y_test_labels, 3)

            # GNB
            self.load_pretrained_models("gaussian_faults_grid_model", X_train_scaled, y_train, X_test_scaled, y_test, 3)

            # MLP
            self.load_pretrained_models("mlp_faults_grid_model", X_train_scaled, y_train_labels, X_test_scaled,
                                        y_test_labels, 3)

    def Adult(self, userResponse):
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

        if userResponse is "2":
            ''' KNN CLASSIFICATION'''

            print('Running KNN Classifier\n')
            param_grid = {
                "n_neighbors": np.arange(5, 40),
                "weights": ['uniform', 'distance']
            }

            knn = Neighbors.KNeighborsClassifier()
            self.grid_search_cv(knn, param_grid, X_train, y_train, X_test,
                                y_test, "knearest_Adult_model", cv=3)

            ''' Decision Tree CLASSIFICATION'''

            print('Running Decision Tree Classifier\n')
            param_grid = {
                'max_depth': np.arange(5, 50, 5),
                'max_leaf_nodes': np.arange(5, 50, 5),
                'criterion': ['gini', 'entropy']
            }

            dtc = Tree.DecisionTreeClassifier(random_state=0)
            self.grid_search_cv(dtc, param_grid, X_train, y_train, X_test,
                                y_test, "tree_Adult_model", cv=3)

            ''' SVM CLASSIFICATION'''

            print('Running SVM Classifier\n')
            param_grid = {
                'kernel': ['rbf'],
                'C': np.logspace(0, 3, 4),
                'gamma': np.logspace(-2, 1, 4)
            }
            svm = sklearn.svm.SVC(random_state=0)
            self.random_search_cv(svm, param_grid, X_train, y_train, X_test, y_test,
                                  "svm_Adult_model", cv=3)

            '''RANDOM FOREST CLASSIFIER'''

            print('Running Random Forest Classifier\n')
            param_grid = {'n_estimators': np.arange(5, 20, 5),
                          'max_depth': np.arange(5, 20,5),
                          'max_leaf_nodes': np.arange(5, 20, 5),
                          'criterion': ['gini', 'entropy']
                          }
            rfc = Ensemble.RandomForestClassifier(random_state=0)
            self.random_search_cv(rfc, param_grid, X_train,
                                  y_train, X_test, y_test, "random_forest_Adult_model", cv=3)

            '''ADABOOST CLASSIFIER'''

            print('Running Adaboost Classifier\n')
            param_grid = {'n_estimators': np.arange(25, 75, 5),
                          'learning_rate': np.arange(0.1, 1.1, 0.1),
                          'algorithm': ['SAMME', 'SAMME.R']
                          }
            adaboost = Ensemble.AdaBoostClassifier(random_state=0)
            self.random_search_cv(adaboost, param_grid, X_train,
                                  y_train, X_test, y_test, "adaboost_Adult_model", cv=3)

            '''LOGISTIC REGRESSION CLASSIFIER'''

            print('Running Logistic Regression Classifier\n')
            param_grid = {
                'C': np.logspace(0, 3, 4),
                'fit_intercept': [True, False],
                'max_iter': [50, 100, 150],
                'solver': ['liblinear', 'sag', 'saga']
            }

            lr = Linear.LogisticRegression(multi_class='auto', random_state=0)
            self.random_search_cv(lr, param_grid, X_train,
                                  y_train, X_test, y_test, "logistic_Adult_model", cv=3)

            '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

            print('Running Gaussian Naive Bayes Classifier\n')
            param_grid = {
                "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test,
                                "gaussian_Adult_model", cv=3)

            '''Neural Network Classifier'''

            mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9,
                                                       learning_rate='adaptive', random_state=0, verbose=True,
                                                       warm_start=True, early_stopping=True)

            param_grid = {
                "solver": ['adam'],
                "learning_rate_init": np.arange(0.1, 1.1, 0.1),
                "hidden_layer_sizes": [(128,), (128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
            }

            self.random_search_cv(mlp, param_grid, X_train, y_train, X_test, y_test, "mlp_Adult_model", cv=3)
        else:
            # SVM
            self.load_pretrained_models("svm_Adult_modelModel", X_train, y_train, X_test, y_test)

            # DTC
            self.load_pretrained_models("tree_Adult_modelModel", X_train, y_train, X_test, y_test)

            # RFC
            self.load_pretrained_models("random_forest_Adult_modelModel", X_train, y_train, X_test, y_test)

            # LR
            self.load_pretrained_models("logistic_Adult_modelModel", X_train, y_train, X_test, y_test)

            # Adaboost
            self.load_pretrained_models("adaboost_Adult_modelModel", X_train, y_train, X_test, y_test)

            # KNN
            self.load_pretrained_models("knearest_Adult_modelModel", X_train, y_train, X_test, y_test)

            # GNB
            self.load_pretrained_models("gaussian_Adult_modelModel", X_train, y_train, X_test, y_test)

            # MLP
            self.load_pretrained_models("mlp_Adult_modelModel", X_train, y_train, X_test, y_test)

    def Yeast(self, userResponse):
        print('Running classification for 8.Yeast dataset')

        df = pd.read_csv("../Datasets/yeast.data", header=None, delim_whitespace=True)
        encoder = Preprocessing.LabelEncoder()
        encoder.fit(df.iloc[:, 0])
        df.iloc[:, 0] = encoder.transform(df.iloc[:, 0])
        encoder.fit(df.iloc[:, 9])
        df.iloc[:, 9] = encoder.transform(df.iloc[:, 9])
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:9], df.iloc[:, 9], test_size=0.2,
                                                            random_state=0, shuffle=False)
        scaler = Preprocessing.StandardScaler().fit(X_train.iloc[:, [0]])
        X_train.iloc[:, [0]] = scaler.fit_transform(X_train.iloc[:, [0]])
        X_test.iloc[:, [0]] = scaler.fit_transform(X_test.iloc[:, [0]])

        if userResponse is "2":

            # LOGISTIC REGRESSION
            param = {'solver': ["lbfgs"],
                     'C': Stats.reciprocal(0.001, 1000),
                     }
            lr = Linear.LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000)
            self.random_search_cv(lr, param, X_train, y_train, X_test, y_test, "YeastLR", 5)

            # K NEAREST NEIGHBOURS
            param = {
                "n_neighbors": [10, 50, 100],
                "weights": ['uniform', 'distance'],
                "leaf_size": [15, 30, 50, 100]
            }
            self.grid_search_cv(Neighbors.KNeighborsClassifier(),
                                param, X_train, y_train, X_test, y_test, "YeastKNN", 5)

            # SVM
            param = {'kernel': ['rbf', 'linear'],
                     'degree': [1, 2, 3, 4, 5, 6],
                     'C': [1, 10, 100, 1000],
                     'gamma': [1e-3, 1e-4]}
            self.grid_search_cv(sklearn.svm.SVC(random_state=0, class_weight="balanced"), param, X_train, y_train,
                                X_test, y_test, "YeastSVM", 5)

            # DECISION TREE CLASSIFIER
            dt = Tree.DecisionTreeClassifier(random_state=0, class_weight="balanced")
            param = {'max_depth': np.arange(3, 10),
                     'criterion': ['gini'],
                     'max_leaf_nodes': [5, 10, 20, 100],
                     'min_samples_split': [2, 5, 10, 20]}
            self.grid_search_cv(dt, param, X_train, y_train, X_test, y_test, "YeastDCT", 5)

            # RANDOM FOREST
            param_grid = {'max_depth': np.arange(3, 10),
                          'max_features': np.arange(1, 9, 1),
                          'max_leaf_nodes': [5, 10, 15, 50, 100],
                          'min_samples_split': [2, 5],
                          }
            self.random_search_cv(Ensemble.RandomForestClassifier(n_estimators=500, random_state=0),
                                  param_grid, X_train, y_train, X_test, y_test, "YeastRF", 5)

            # ADABOOST CLASSIFIER
            param_grid = {
                "n_estimators": [30, 50, 70, 100],
                "learning_rate": [0.5, 0.7, 1, 2],
                "algorithm": ["SAMME", "SAMME.R"]
            }
            self.grid_search_cv(Ensemble.AdaBoostClassifier(random_state=0), param_grid, X_train,
                                y_train, X_test, y_test, "YeastAda", 3)

            # NEURAL NETWORK
            param_grid = {
                "solver": ['adam', 'sgd'],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "hidden_layer_sizes": [(1024,), (128, 64, 32), (512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
            }
            self.grid_search_cv(
                NN.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10, momentum=0.9, \
                                 learning_rate='adaptive', verbose=True, warm_start=True, \
                                 early_stopping=True), param_grid, X_train, y_train, X_test, y_test, "YeastNN", 3)

            # GAUSSIAN NAIVE BAY
            param_grid = {
                "var_smoothing": [1e-07, 1e-08, 1e-09]
            }
            self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test, "YeastGNN", 5)
        else:
            # ADABOOST
            self.load_pretrained_models("YeastAdaModel", X_train, y_train, X_test, y_test)

            # DTC
            self.load_pretrained_models("YeastGNNModel", X_train, y_train, X_test, y_test)

            # KNN
            self.load_pretrained_models("YeastKNNModel", X_train, y_train, X_test, y_test)

            # LR
            self.load_pretrained_models("YeastLRModel", X_train, y_train, X_test, y_test)

            # NN
            self.load_pretrained_models("YeastNNModel", X_train, y_train, X_test,y_test)

            # RF
            self.load_pretrained_models("YeastRFModel", X_train, y_train, X_test, y_test)

            # SVM
            self.load_pretrained_models("YeastSVMModel", X_train, y_train, X_test, y_test)

            # DCT
            self.load_pretrained_models("YeastDCTModel", X_train, y_train, X_test, y_test)

    def Thoracic_Surgery_Data(self, userResponse):
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

            if userResponse is "2":

                ''' KNN CLASSIFICATION'''

                print('Running KNN Classifier\n')
                param_grid = {
                    "n_neighbors": np.arange(5, 50),
                    "weights": ['uniform', 'distance'],
                    "leaf_size": np.arange(5, 100, 5)
                }
                knn = Neighbors.KNeighborsClassifier()
                self.grid_search_cv(knn, param_grid, X_train, y_train, X_test,
                                    y_test, "knearest_Thoracic_Surgery_Data_model", cv=3)

                ''' Decision Tree CLASSIFICATION'''

                print('Running Decision Tree Classifier\n')
                param_grid = {
                    'max_depth': np.arange(5, 30, 5),
                    'max_leaf_nodes': np.arange(5, 30, 5),
                    'criterion': ['gini', 'entropy']
                }
                dtc = Tree.DecisionTreeClassifier(random_state=0)

                self.grid_search_cv(dtc, param_grid, X_train, y_train, X_test,
                                    y_test, "tree_Thoracic_Surgery_Data_model", cv=3)

                ''' SVM CLASSIFICATION'''

                print('Running SVM Classifier\n')
                param_grid = {
                    'kernel': ['rbf'],
                    'C': np.logspace(0, 3, 4),
                    'gamma': np.logspace(-2, 1, 4)
                }
                svm = sklearn.svm.SVC(random_state=0)
                self.random_search_cv(svm, param_grid, X_train, y_train, X_test, y_test,
                                      "svm_Thoracic_Surgery_Data_model", cv=3)

                '''RANDOM FOREST CLASSIFIER'''

                print('Running Random Forest Classifier\n')
                param_grid = {'n_estimators': np.arange(5, 20, 3),
                              'max_depth': np.arange(5, 50, 3),
                              'max_leaf_nodes': np.arange(5, 50, 5),
                              'criterion': ['gini', 'entropy']
                              }

                rfc = Ensemble.RandomForestClassifier(random_state=0)

                self.grid_search_cv(rfc, param_grid, X_train,
                                    y_train, X_test, y_test, "random_forest_Thoracic_Surgery_Data_model", cv=3)

                '''ADABOOST CLASSIFIER'''

                print('Running Adaboost Classifier\n')
                param_grid = {'n_estimators': np.arange(25, 75, 5),
                              'learning_rate': np.arange(0.1, 1.1, 0.1),
                              'algorithm': ['SAMME', 'SAMME.R']
                              }

                adaboost = Ensemble.AdaBoostClassifier(random_state=0)

                self.random_search_cv(adaboost, param_grid, X_train,
                                      y_train, X_test, y_test, "adaboost_Thoracic_Surgery_Data_model", cv=3)

                '''LOGISTIC REGRESSION CLASSIFIER'''

                print('Running Logistic Regression Classifier\n')
                param_grid = {
                    'C': np.logspace(0, 3, 4),
                    'fit_intercept': [True, False],
                    'max_iter': [50, 100, 150],
                    'solver': ['liblinear', 'sag', 'saga']
                }
                lr = Linear.LogisticRegression(multi_class='auto', random_state=0)
                self.random_search_cv(lr, param_grid, X_train,
                                      y_train, X_test, y_test, "logistic_Thoracic_Surgery_Data_model", cv=3)

                '''GAUSSIAN NAIVE BAYES CLASSIFIER'''

                print('Running Gaussian Naive Bayes Classifier\n')
                param_grid = {
                    "var_smoothing": [1e-05, 1e-07, 1e-09, 1e-11]}
                self.grid_search_cv(GaussianNB(), param_grid, X_train, y_train, X_test, y_test,
                                    "gaussian_Thoracic_Surgery_Data_model", cv=3)

                '''Neural Network Classifier'''

                mlp = sklearn.neural_network.MLPClassifier(activation='relu', tol=1e-4, n_iter_no_change=10,
                                                           momentum=0.9,
                                                           learning_rate='adaptive', random_state=0, verbose=True,
                                                           warm_start=True, early_stopping=True)

                param_grid = {
                    "solver": ['adam'],
                    "learning_rate_init": np.arange(0.1, 1.1, 0.1),
                    "hidden_layer_sizes": [(512,), (256, 128, 64, 32, 2), (512, 256, 128, 64, 32, 2)]
                }

                self.random_search_cv(mlp, param_grid, X_train, y_train, X_test, y_test,
                                      "mlp_Thoracic_Surgery_Data_model", cv=3)
            else:
                # SVM
                self.load_pretrained_models("svm_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

                # DTC
                self.load_pretrained_models("tree_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

                # RFC
                self.load_pretrained_models("random_forest_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test,
                                            y_test)

                # LR
                self.load_pretrained_models("logistic_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

                # Adaboost
                self.load_pretrained_models("adaboost_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

                # KNN
                self.load_pretrained_models("knearest_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

                # GNB
                self.load_pretrained_models("gaussian_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

                # MLP
                self.load_pretrained_models("mlp_Thoracic_Surgery_Data_modelModel", X_train, y_train, X_test, y_test)

    def Seismic_Bumps(self, userResponse):
        print('Running classification for 10.Seismic Bumps dataset')
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

        if userResponse is "2":
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
            self.load_pretrained_models("svm_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled, y_test, 3)

            # DTC
            self.load_pretrained_models("tree_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled, y_test, 3)

            # RFC
            self.load_pretrained_models("random_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled, y_test,
                                        3)

            # LR
            self.load_pretrained_models("logistic_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled,
                                        y_test, 3)

            # Adaboost
            self.load_pretrained_models("adaboost_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled,
                                        y_test, 3)

            # KNN
            self.load_pretrained_models("kNearest_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled,
                                        y_test, 3)

            # GNB
            self.load_pretrained_models("gaussian_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled,
                                        y_test, 3)

            # MLP
            self.load_pretrained_models("mlp_bumps_grid_model", X_smote_train, y_smote_train, X_test_scaled, y_test, 3)
