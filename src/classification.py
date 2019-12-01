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

class class_classification:
    '''Contains all the classifiers'''


    def grid_search_cv(self, classifier, param_grid, X, y):
        model = model_select.GridSearchCV(classifier, param_grid, cv=3, verbose=10).fit(X,y)
        '''valida
        lear
        confs'''

    def random_search_cv(self,classifier,param_grid,X,y):
        model = model_select.RandomizedSearchCV(classifier,param_grid,cv=3,verbose=10).fit(X,y)

    def run_classifier(self):
        print('Running classifiers for the following datasets: \n')
        self.Diabetic_Retinopathy()
        self.Default_of_credit_card_clients()
        self.Breast_Cancer_Wisconsin()
        self.Statlog_Australian()
        self.Statlog_German()
        self.Steel_Plates_Faults()
        self.Adult()
        self.Yeast()
        self.Thoracic_Surgery_Data()
        self.Seismic_Bumps()

    def Diabetic_Retinopathy(self):
        print('Running classification for 1.Diabetic Retinopathy dataset')

    def Default_of_credit_card_clients(self):
        print('Running classification for 2.Default of credit card clients dataset')

    def Breast_Cancer_Wisconsin(self):
        print('Running classification for 3.Breast Cancer Wisconsin dataset')


        ''' DATASET WBC'''


        df_wbc = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
            delimiter=",", header=None,names=['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                   'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

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
                         delimiter=",", header=None, names=['id', 'Diagnosis','radius', 'texture', 'perimeter', 'area', 'smoothness',
                                                            'compactness', 'concavity', 'concave points', 'symmetry',
                                                            'fractal dimension','radius SE', 'texture SE', 'perimeter SE', 'area SE',
                                                            'smoothness SE', 'compactness SE', 'concavity SE','concave points SE',
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
                                                            'worst fractal dimension', 'Tumor size', 'Lymph node status'])

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




