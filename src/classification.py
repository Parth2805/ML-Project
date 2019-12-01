import arff
import sklearn.model_selection as model_select
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.svm as SVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, GridSearchCV, RandomizedSearchCV, learning_curve
import numpy as np
import sklearn.tree as Tree
import pickle
import sklearn.ensemble as Ensemble
import scipy.stats as Stats
import sklearn.linear_model  as Linear
from sklearn.naive_bayes import GaussianNB
import sklearn.neural_network as NN
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from scipy.io import arff


class class_classification:
    '''Contains all the classifiers'''

    def grid_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test,cv=5):
        model = model_select.GridSearchCV(classifier, param_grid, verbose=10,cv=cv).fit(X_train, y_train)

        '''valida
        lear
        confs'''

    def random_search_cv(self, classifier, param_grid, X_train, y_train, X_test, y_test,cv=5):
        model = model_select.GridSearchCV(classifier, param_grid, cv=cv, verbose=10).fit(X_train, y_train)

        # print(model.best_estimator_.score(X_test, y_test))

    def run_classifier(self):
        print('Running classifiers for the following datasets: \n')
        self.Diabetic_Retinopathy()
        # self.Default_of_credit_card_clients()
        # self.Breast_Cancer_Wisconsin()
        # self.Statlog_Australian()
        # self.Statlog_German()
        # self.Steel_Plates_Faults()
        # self.Adult()
        # self.Yeast()
        # self.Thoracic_Surgery_Data()
        # self.Seismic_Bumps()

    def Diabetic_Retinopathy(self):
        '''Preprocessing'''

        print('Running classification for 1.Diabetic Retinopathy dataset')

        file = "C:/Users/parth's alienware/PycharmProjects/ML6321/Datasets/1_DiabeticRetinopathy.arff"
        df, metadata = arff.loadarff(file)

        data = pd.DataFrame(df)
        data = data.values
        data[:, 19] = np.where(data[:, 19] == b'0', 0, data[:, 19])
        data[:, 19] = np.where(data[:, 19] == b'1', 1, data[:, 19])
        # data=data.astype(float)
        # print(data)

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

        self.random_search_cv(lr, param, X_train, y_train, X_test, y_test)

        # %%
        '''
        ### K-**Neighbors**
        '''
        # %%

        k_n = sklearn.neighbors.KNeighborsClassifier()

        param = dict(weights=['uniform', 'distance'],
                     n_neighbors=[5, 10, 15, 20, 50, 100, 200, 500])

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

    def Default_of_credit_card_clients(self):
        print('Running classification for 2.Default of credit card clients dataset')

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

    def Statlog_Australian(self):
        print('Running classification for 4.Statlog Australian dataset')

    def Statlog_German(self):
        print('Running classification for 5.Statlog German dataset')

    def Steel_Plates_Faults(self):
        print('Running classification for 6.Steel Plates Faults dataset')

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

    def Seismic_Bumps(self):
        print('Running classification for 10.Seismic Bumps dataset')
