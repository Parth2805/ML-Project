import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn
import sklearn.metrics as metrics
import sklearn.neighbors
import sklearn.neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class class_regression:
    '''Contains all the regression logic'''

    def grid_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test, cv=5):
        model = model_select.GridSearchCV(classifier, param_grid, cv=cv, verbose=10).fit(X_train, y_train)

        # model.best_estimator_.score(X_test, y_test)

    def random_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test, cv=5):
        model = model_select.GridSearchCV(classifier, param_grid, cv=cv, verbose=10).fit(X_train, y_train)

        # model.best_estimator_.score(X_test, y_test)

    def get_regressor(self):
        print('Running regressors for the following datasets: \n')
        self.WineQuality()
        self.Communities_Crime()
        self.QSAR_aquatic_toxicity()
        self.Parkinson_Speech()
        self.Facebook_metrics()
        self.Bike_Sharing()
        self.Student_Performance()
        self.Concrete_Compressive_Strength()
        self.SGEMM_GPU_kernel_performance()
        self.Merck_Molecular_Activity_Challenge()

    def WineQuality(self):
        print('Running Regression for 1.WineQuality dataset')

        # %%
        '''
        ### **Preprocessing**
        '''

        # %%
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

        '''
        ### **Linear Regression**
        '''
        lr = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
        # print("Validation Score:",lr.score(X_train,y_train))
        # print("Testing Score",lr.score(X_test,y_test))

        '''
        ### **SVR**
        '''
        svm = sklearn.svm.SVR()

        param = [{"kernel": ["rbf"],
                  "degree": [1, 2, 3],
                  "C": np.array([1, 2, 5, 10, 20]),
                  "gamma": np.array([0.1, 1, 5, 10])}]

        '''
        ### **Decision Tree**
        '''
        dt = sklearn.tree.DecisionTreeRegressor(random_state=0)

        param = {'max_depth': np.arange(1, 15, 1),
                 'splitter': ['best', 'random'],
                 'max_features': np.arange(1, 11, 1),
                 'min_samples_split': np.arange(2, 20, 1)}

        '''
        ### **Random Forest**
        '''

        rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=0)

        param = {'max_depth': np.arange(1, 20, 1),
                 'max_features': np.array([1, 2, 5, 11]),
                 'min_samples_split': np.array([2, 3, 5])}

        '''
        ## **Ada Boost**
        '''

        ada = sklearn.ensemble.AdaBoostRegressor(random_state=0)

        param = dict(n_estimators=np.arange(50, 250, 10),
                     loss=['linear', 'square']
                     )

        '''
        ## **Neural Network**
        '''
        mlp = sklearn.neural_network.MLPRegressor(activation='relu', n_iter_no_change=10, momentum=0.9,
                                                  learning_rate='adaptive', random_state=0, verbose=True,
                                                  warm_start=True, early_stopping=True, )

        param_grid = {
            "solver": ['adam'],
            "learning_rate_init": reciprocal(0.001, 0.1),
            "hidden_layer_sizes": [(512,), (256, 128, 64, 32), (512, 256, 128, 64, 32)]
        }


        '''
        ## **Gaussian Process**
        '''
        gp = sklearn.gaussian_process.GaussianProcessRegressor(random_state=0)

        param_grid = {

            "n_restarts_optimizer": np.arange(0, 10, 1)
        }




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

    def Facebook_metrics(self):
        print('Running Regression for 5.Facebook_metrics dataset')

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

    def Concrete_Compressive_Strength(self):
        print('Running Regression for 8.Concrete_Compressive_Strength dataset')

    def SGEMM_GPU_kernel_performance(self):
        print('Running Regression for 9.SGEMM_GPU_kernel_performance dataset')

    def Merck_Molecular_Activity_Challenge(self):
        print('Running Regression for 10.Merck_Molecular_Activity_Challenge dataset')
