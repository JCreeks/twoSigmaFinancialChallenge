
# coding: utf-8

# In[6]:

from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import log_loss, mean_squared_error as mse, r2_score 
from sklearn.metrics.scorer import make_scorer


# In[7]:

def runXGB(X_train, y_train, X_test=None, num_class=1, feature_names=None, seed=0, num_rounds=1000, early_stopping_rounds=None):
    params = {
    'booster': 'gbtree',
    'objective': 'reg:linear', #'multi:softprob'
    'subsample': 0.8,
    'colsample_bytree': 0.85, #like max_features
    'eta': 0.05,
    'max_depth': 7,
    'seed': seed,
    'silent': 0,
    'eval_metric': 'rmse' # "logloss", "mlogloss", auc" # for ranking 
    }
    
    if num_class!=1:
        params['num_class']=num_class

    plst = list(params.items())
    dtrain = xgb.DMatrix(X_train, y_train)
    
    model = xgb.train(plst, dtrain, num_boost_round=num_rounds, early_stopping_rounds=early_stopping_rounds)

    if X_test is not None:
        dtest = xgb.DMatrix(X_test)
        pred = model.predict(dtest)
        return pred, model
    return None, model


# In[8]:

def runXGBShuffle(X_train, y_train, X_test, num_class=1, feature_names=None, seed=0, num_rounds=1000, test_size=.3,                early_stopping_rounds=None):
    params = {
    'booster': 'gbtree',
    'objective': 'reg:linear', #'multi:softprob'
    'subsample': 0.8,
    'colsample_bytree': 0.85, #like max_features
    'eta': 0.05,
    'max_depth': 7,
    'seed': seed,
    'silent': 0,
    'eval_metric': 'rmse' # "logloss", "mlogloss", auc" # for ranking 
    }
    
    if num_class!=1:
        params['num_class']=num_class

    plst = list(params.items())
    X_dtrain, X_deval, y_dtrain, y_deval=train_test_split(X_train, y_train, random_state=seed, test_size=test_size)
    dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
    deval = xgb.DMatrix(X_deval, y_deval)
    watchlist = [(deval, 'eval')]
    
    model = xgb.train(plst, dtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds)

    if X_test is not None:
        dtest = xgb.DMatrix(X_test)
        pred = model.predict(dtest)
        return pred, model
    return None, model


# In[43]:

class xgbClass(object):
    def __del__(self):
        return 
    def __init__(self, eta=.1, subsample=.8, num_class=1, max_depth=5, seed=17, silent=0, eva_metric='mlogloss',                colsample_bytree=1, objective='solfprob', min_child_weight=1, num_rounds=500, early_stopping_rounds=None):
        self.params={
        'objective' : objective, #'reg:linear','multi:softprob'
        'subsample' : subsample,
        'colsample_bytree' : colsample_bytree, #like max_features
        'eta': eta,
        'max_depth': max_depth,
        'seed': seed,
        'silent': silent,
        'eval_metric': eva_metric, #'rmse' "logloss", "mlogloss", auc" # for ranking
        'min_child_weight': min_child_weight
        }
        self.num_rounds=num_rounds
        self.early_stopping_rounds=early_stopping_rounds
        
        if num_class!=1:
            self.params['num_class']=num_class
        self.model=[]
        
    def fit(self, X_train, y_train):#, early_stopping_rounds=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_rounds, #early_stopping_rounds=early_stopping_rounds)
                               early_stopping_rounds=self.early_stopping_rounds)
    
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
    
    def predict_proba(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)


def modelfit(alg, dtrain, predictors, target, outputMetrics=mse, XGBmetrics='rmse', useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=XGBmetrics, early_stopping_rounds=early_stopping_rounds)#, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric=XGBmetrics)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "CV metrics : %.4g" % outputMetrics(dtrain[target].values, dtrain_predictions)
    #print metrics+" Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')