# coding: utf-8

# In[1]:

from sklearn.grid_search import GridSearchCV
import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from time import time
import xgboost as xgb
import itertools


# In[2]:

env = kagglegym.make()
o = env.reset()
#o.train = o.train[:1000]
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]

#train = pd.read_hdf(r'C:\Users\jiguo\Desktop\KProject\input\train.h5')
O = pd.read_hdf('../input/train.h5')
d_mean= O[col].median(axis=0)
ymean_dict = dict(o.train.groupby(["id"])["y"].median())



def NAOperation(train, d_mean):
    n = train.isnull().sum(axis=1)
    for c in train.columns:
        train[c + '_nan_'] = pd.isnull(train[c])
        d_mean[c + '_nan_'] = 0
    #train = train.fillna(d_mean)
    train['znull'] = n
    n = []
    return train


class RTR(object):
    def __init__(self, d_mean, n_estimators=40, max_depth=4, n_jobs=-1,                                                         random_state=17, verbose=0):
#         self.objective=objective
#         self.colsample_bytree=colsample_bytree
#         self.subsample=subsample
#         self.min_child_weight=min_child_weight
#         self.base_score=base_score
        self.d_mean=d_mean
        self.model=ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,\
                                   n_jobs=n_jobs, random_state=random_state, \
                                   verbose=verbose)
        
    def fit(self, X_train, y_train):
        self.model.fit(NAOperation(X_train, self.d_mean), y_train)
        
    def predict(self, X_test):
        return self.model.predict(NAOperation(X_test, self.d_mean))

class LR_tech_20():
    def __init__(self, d_mean, col, low_y_cut=-0.085, high_y_cut=0.075, n_jobs=-1):
        self.low_y_cut = low_y_cut
        self.high_y_cut = high_y_cut
        self.model=LinearRegression(n_jobs=n_jobs)
        self.d_mean=d_mean
        self.col=col
        
    def fit(self, x_train, y_train):
        #d_mean= x_train.median(axis=0)
        #x_train=x_train.fillna(d_mean)
        #x_train = x_train[self.col]
        y_is_above_cut = (y_train > self.high_y_cut)
        y_is_below_cut = (y_train < self.low_y_cut)
        y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
        self.model.fit(np.array(x_train.loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), y_train.loc[y_is_within_cut])

    def predict(self, test):
        return self.model.predict(np.array(test[self.col].fillna(self.d_mean)['technical_20'].values).\
        reshape(-1,1)).clip(self.low_y_cut, self.high_y_cut)


# In[8]:

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.S_train=[]
        self.S_test=[]
        self_folds=[]
        
    def fit(self, X, y):
        #X = np.array(X)
        #y = np.array(y)
        self.folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=False, random_state=17))
        self.S_train = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_holdout = X.iloc[test_idx]
                # y_holdout = y[test_idx]
                clf[j].fit(X_train, y_train)
                y_pred = clf[j].predict(X_holdout)[:]
                self.S_train[test_idx, i] = y_pred
        self.stacker.fit(self.S_train, y)
        
    def predict(self, T):       
        #T = np.array(T)       
        self.S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(self.folds)))
            for j in range(len(self.folds)):
                S_test_i[:, j] = clf[j].predict(T)[:]
            self.S_test[:, i] = S_test_i.mean(1)
#             self.S_test[:,i]=[sum(map(lambda x: (x.predict(T))[0],clf))/self.n_folds]
        y_pred = self.stacker.predict(self.S_test)[:]
        return y_pred


# In[ ]:

start=time()
#model1 = ExtraTreesRegressor(n_estimators=4, max_depth=4, n_jobs=7, random_state=17, verbose=0)
#model2 = LR_tech_20(d_mean=d_mean, col=col, n_jobs=7)
n_folds = 2
ensembleObj=Ensemble(n_folds=n_folds, stacker=LinearRegression(fit_intercept=False, n_jobs=-1), \
base_models=[[RTR(n_estimators=50, max_features='auto', max_depth=4, n_jobs=-1, random_state=17, \
                  verbose=0) for i in range(n_folds)], \
[LR_tech_20(d_mean=d_mean, col=col, n_jobs=-1) for i in range(n_folds)], \
[xgb.XGBRegressor(objective='reg:linear', colsample_bytree=.8, subsample=.8, min_child_weight=1000,\
base_score=.5) for i in range(n_folds)]])
ensembleObj.fit(X=o.train[col], y=o.train.y) #x is filled na
end = time()
print(end - start)


# In[ ]:

print(ensembleObj.stacker.coef_)


# In[ ]:

start=time()

coeff1=.975
while True:
    test = o.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
         test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    pred = o.target
    pred['y'] = ensembleObj.predict(T=test)
    pred['y'] = pred.apply(lambda r: coeff1 * r['y'] +(1-coeff1) * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
        
end = time()
print(end - start)