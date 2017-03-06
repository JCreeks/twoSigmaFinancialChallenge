
# coding: utf-8

# In[7]:

from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import log_loss, mean_squared_error as mse, r2_score 
from sklearn.metrics.scorer import make_scorer


# In[18]:

def CVScore(model, X_train, y_train, n_splits=5, is_TimeSeries=False, seed=2017, my_score=mse):
    cv_scores = []
    if not len(np.array(X_train).shape)==0:
        X_train=np.array(X_train)
        y_train=np.array(y_train)
    if not is_TimeSeries:
        kf=KFold(n_splits=n_splits, shuffle=True, random_state=17)
    else:
        kf=TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in kf.split(X_train):
        X_CVtrain = X_train[train_idx]
        y_CVtrain = y_train[train_idx]
        X_CVholdout = X_train[test_idx]
        y_CVholdout = y_train[test_idx]
        model.fit(X_CVtrain, y_CVtrain)
        if my_score==log_loss:
            pred=model.predict_proba(X_CVholdout)
        else:
            pred = model.predict(X_CVholdout)[:]
        cv_scores.append(my_score(y_CVholdout, pred))
    return np.mean(cv_scores)






