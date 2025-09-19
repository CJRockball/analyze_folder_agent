#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time 
from datetime import datetime
import pytz

import gc

# local time
local_tz = pytz.timezone('Asia/Singapore')

import lightgbm as lgb
import xgboost as xgb

from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso

#Optuna settings
import optuna
import logging
from tqdm.auto import tqdm

## Config level of msgs
logging.getLogger("optuna").setLevel(logging.WARNING)

## Callback for process bar
def tqdm_callback(study, trial):
    global pbar
    pbar.update(1)

random_state = 42

#%%


df_train = pd.read_csv('data/raw/train.csv')
df_train = df_train.drop(columns=['id'])
df_org = pd.read_csv('data/raw/credit_risk_dataset.csv')

df_X = pd.concat([df_train, df_org], axis=0).reset_index(drop=True)

X_cols = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length']

df_X = df_X.loc[~df_X.duplicated(X_cols),:].reset_index(drop=True)
df_oof = pd.DataFrame(index=df_X.index)

df_y = df_X.pop('loan_status')

df_test = pd.read_csv('data/raw/test.csv')
df_test = df_test.drop(columns=['id'])

# Determine the numerical variables
TARGET = ['loan_status']
NUMS = [col for col in df_X.columns if df_X[col].dtype in ['int64', 'float64']]
# Determine the categorical variables
CATS = [col for col in df_X.columns if df_X[col].dtype == 'object']

# cat indices
cat_indices = [df_X.columns.get_loc(col) for col in CATS]

#%%

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_oe = pd.DataFrame(oe.fit_transform(df_X[CATS]), columns = CATS).fillna(-1).astype(int)
test_oe = pd.DataFrame(oe.transform(df_test[CATS]), columns = CATS).fillna(-1).astype(int)

X_oe_concated = pd.concat([X_oe, df_X[NUMS]], axis= 1)
test_oe_concated = pd.concat([test_oe, df_test[NUMS]], axis= 1)

#%%

xgb_params= {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'eta': 0.07964177396162775,
    'reg_lambda': 38.499443612904315,
    'subsample': 0.8778759317150353,
    'colsample_bytree': 0.6504220261795185,
    'random_state': random_state,
    'verbosity':0,
    'eneable_categorical': True,
    'min_child_weight': 5,
    #'tree_method': 'hist',
}

#%%

rkf = RepeatedKFold(n_splits=15, n_repeats=1, random_state=random_state)

def cross_val_xgb(X, y, kf, test, model_name="Model"):

    start= time.time()

    folds = kf # definir antes
    test_preds = np.zeros(len(test))
    oof_preds = np.zeros(len(y))
    y_vals = np.zeros(len(y))
    scores=[]

    for n_fold, (tr, val) in enumerate(kf.split(X, y)):
        # 
        X_train_, X_test_, y_train_, y_test_ = X.iloc[tr, :].to_numpy(), X.iloc[val, :].to_numpy(), y.iloc[tr].to_numpy(), y.iloc[val].to_numpy()
        
        dtrain = xgb.DMatrix(X_train_, label=y_train_)
        dvalid = xgb.DMatrix(X_test_, label=y_test_)
        
        eval_set = [(dtrain, 'train'), (dvalid, 'valid')]

        current_time = datetime.now(pytz.utc).astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f'{model_name} - Fitting Fold {n_fold + 1} started at {current_time}')

        xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1500,
        #feval=rmse_metric,
        maximize=True,
        evals=eval_set,
        early_stopping_rounds=150,
        verbose_eval=200
        )
        
        y_pred_val = xgb_model.predict(dvalid)
        oof_preds[val] = y_pred_val
        y_vals[val] = y_test_
        
        score = roc_auc_score(y_test_, y_pred_val)

        scores.append(score)
        dtest_data = xgb.DMatrix(test.values)
        test_preds += xgb_model.predict(dtest_data) / folds.get_n_splits()

        print(f'{model_name} - Fold {n_fold + 1} ROC_AUC score : {score:.5f}')
        print('-' * 50)
        
        del dtrain, dvalid, X_train_, y_train_, X_test_, y_test_, xgb_model
        gc.collect()

    # Results    
    print(f'{model_name} - Mean ROC_AUC score +/- std. dev.: '
        f'{np.array(scores).mean():.5f} +/- {np.array(scores).std():.3f}')
    print(f'\n{model_name} - Time spent[s]: {(time.time()-start)/60:.2f} minutes')


    return oof_preds, y_vals, test_preds, scores

#%%

preds_train, preds_test, cv_summary = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#%%

# Cross-val XGBoost
preds_train['XGBoost'], _, preds_test['XGBoost'], cv_summary['XGBoost'] = \
    cross_val_xgb(X_oe_concated, df_y, rkf, test_oe_concated, model_name= 'XGBoost')

#%%
df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['loan_status'] = preds_test['XGBoost'].values

df_sub.to_csv('submissions/cat_35.csv', index=False)
df_sub.to_csv('ensembling2/35_pred.csv')
df_check = pd.read_csv('submissions/cat_35.csv')
display(df_check)

df_oof['loan_status'] = preds_train['XGBoost'].values
display(df_oof)
df_oof.to_csv('ensembling2/35_oof.csv')






# %%
