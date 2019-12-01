import sklearn.model_selection as MS
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


    def grid_search_cv(self, modelType, param_grid, X, y):
        model = MS.GridSearchCV(modelType, param_grid, cv=3, verbose=10).fit(X,y)
        '''valida
        lear
        confs'''

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



        '''DATASET WPBC'''

    def Statlog_Australian(self):
        print('Running classification for 4.Statlog Australian dataset')

    def Statlog_German(self):
        print('Running classification for 5.Statlog German dataset')

    def Steel_Plates_Faults(self):
        print('Running classification for 6.Steel Plates Faults dataset')

    def Adult(self):
        print('Running classification for 7.Adult dataset')

    def Yeast(self):
        print('Running classification for 8.Yeast dataset')

    def Thoracic_Surgery_Data(self):
        print('Running classification for 9.Thoracic Surgery Data dataset')

    def Seismic_Bumps(self):
        print('Running classification for 10.Seismic Bumps dataset')




