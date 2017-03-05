
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
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    if not is_TimeSeries:
        kf=KFold(n_splits=n_splits, shuffle=True, random_state=17)
    else:
        kf=TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in kf.split(np.arange(len(X_train))):
        X_CVtrain = X_train[train_idx]
        y_CVtrain = y_train[train_idx]
        X_CVholdout = X_train[test_idx]
        y_CVholdout = y_train[test_idx]
        model.fit(X_CVtrain, y_CVtrain)
        pred = model.predict(X_CVholdout)[:]
        cv_scores.append(my_score(y_CVholdout, pred))
    return np.mean(cv_scores)


# In[31]:

n=1000
np.random.seed(17)
X=pd.DataFrame(np.random.randn(n,1))
y=X.iloc[:,0]+.2*pd.Series(np.random.randn(n))
X_train,y_train=X.iloc[:n/2], y.iloc[:n/2]
X_test, y_test=X.iloc[n/2:], y.iloc[n/2:]


# In[32]:

CVScore(linear_model.LinearRegression(fit_intercept=False), X_train, y_train)


# In[ ]:



