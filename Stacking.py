
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
#from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit
from XGBoostPackage import xgbClass


# In[29]:

class Ensemble(object):
    #base_models=[model1, model2, model3,...]
    def __init__(self, n_folds, stacker, base_models, is_TimeSeries=False):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.S_train=[]
        self.S_test=[]
        self.isTS=is_TimeSeries
        #self_folds=[]
        
    def fit(self, X_train, y_train):
        if not len(np.array(X_train).shape)==0:
            X_train=np.array(X_train)
            y_train=np.array(y_train)
        if not self.isTS:
            kf=KFold(n_splits=self.n_folds, shuffle=True, random_state=17)
        else:
            kf=TimeSeriesSplit(n_splits=self.n_folds)
        self.S_train = np.zeros((X_train.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            #clf=[clfModel for tmp in np.arange(self.n_folds)]
            for j, (train_idx, test_idx) in enumerate(kf.split(X_train)):
                print('startig training model {}, training set {}'.format(i,j))
                X_CVtrain = X_train[train_idx]
                y_CVtrain = y_train[train_idx]
                X_CVholdout = X_train[test_idx]
                # y_CVholdout = y_train[test_idx]
                clf[j].fit(X_CVtrain, y_CVtrain)
                y_pred = clf[j].predict(X_CVholdout)[:]
                self.S_train[test_idx, i] = y_pred
        self.stacker.fit(self.S_train, y_train)
        
    def predict(self, X_test):    
        if not len(np.array(X_test).shape)==0:
            X_test=np.array(X_test)
        self.S_test = np.zeros((X_test.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X_test.shape[0], self.n_folds))
            for j in range(self.n_folds):
                print('startig predicting model {}, training set {}'.format(i,j))
                S_test_i[:, j] = clf[j].predict(X_test)[:]
            self.S_test[:, i] = S_test_i.mean(1)
        y_pred = self.stacker.predict(self.S_test)[:]
        return y_pred

class EnsembleClassifier(object):
    #base_models=[model1, model2, model3,...]
    def __init__(self, n_folds, stacker, base_models, is_TimeSeries=False, n_class=1):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.S_train=[]
        self.S_test=[]
        self.isTS=is_TimeSeries
        self.n_class=n_class
        #self_folds=[]
        
    def fit(self, X_train, y_train):
        if not len(np.array(X_train).shape)==0:
            X_train=np.array(X_train)
            y_train=np.array(y_train)
        if not self.isTS:
            kf=KFold(n_splits=self.n_folds, shuffle=True, random_state=17)
        else:
            kf=TimeSeriesSplit(n_splits=self.n_folds)
        self.S_train = np.zeros((X_train.shape[0], len(self.base_models)*self.n_class))
        for i, clf in enumerate(self.base_models):
            #clf=[clfModel for tmp in np.arange(self.n_folds)]
            for j, (train_idx, test_idx) in enumerate(kf.split(X_train)):
                print('startig training model {}, training set {}'.format(i,j))
                X_CVtrain = X_train[train_idx]
                y_CVtrain = y_train[train_idx]
                X_CVholdout = X_train[test_idx]
                # y_CVholdout = y_train[test_idx]
                clf[j].fit(X_CVtrain, y_CVtrain)
                y_pred = clf[j].predict_proba(X_CVholdout)
                self.S_train[test_idx, (i*self.n_class):((i+1)*self.n_class)] = y_pred
        self.stacker.fit(self.S_train, y_train)
        
    def predict(self, X_test):    
        if not len(np.array(X_test).shape)==0:
            X_test=np.array(X_test)
        self.S_test = np.zeros((X_test.shape[0], len(self.base_models)*self.n_class))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X_test.shape[0], self.n_class, self.n_folds))
            for j in range(self.n_folds):
                print('startig predicting model {}, training set {}'.format(i,j))
                S_test_i[:, :, j] = clf[j].predict_proba(X_test)
            self.S_test[:, (i*self.n_class):((i+1)*self.n_class)] = S_test_i.mean(2)
        y_pred = self.stacker.predict(self.S_test)
        return y_pred
    
    def predict_proba(self, X_test):    
        if not len(np.array(X_test).shape)==0:
            X_test=np.array(X_test)
        self.S_test = np.zeros((X_test.shape[0], len(self.base_models)*self.n_class))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X_test.shape[0], self.n_class, self.n_folds))
            for j in range(self.n_folds):
                print('startig predicting model {}, training set {}'.format(i,j))
                S_test_i[:, :, j] = clf[j].predict_proba(X_test)
            self.S_test[:, (i*self.n_class):((i+1)*self.n_class)] = S_test_i.mean(2)
        y_pred = self.stacker.predict_proba(self.S_test)
        return y_pred



