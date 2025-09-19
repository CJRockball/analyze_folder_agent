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
# LGBM params
lgb_params= {
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'verbosity': -1,
    'n_estimators': 1500,
    'max_bin': 1024,
    'boosting_type': 'gbdt', #'dart'
    'colsample_bytree': 0.5673775386473462,        
    'eta': 0.05446876730023387,
    'reg_lambda': 10.787843597294561,
    'min_child_samples': 69,
    'random_state': random_state,
    'early_stopping_rounds': 150,
    'verbose':1,
    #'categorical_feature': cat_indices,
}

#%% Set up optuna search

rkf = RepeatedKFold(n_splits=15, n_repeats=1, random_state=random_state)
SEARCH= False
n_trials = 100
timeout= 3600*3

#%%
# Define el objetivo para Optuna
def objective_lgbm(trial, X, y):
    max_depth = -1
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': 0,
        'max_bin': 1024,
        'boosting_type': 'gbdt', #'gbdt'
        #'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        #'num_leaves': trial.suggest_int('num_leaves', 2 ** (max_depth - 1), 2 ** max_depth),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),        
        'eta': trial.suggest_float('eta', 0.05, 0.1),
        #'lambda_l1': trial.suggest_float("reg_alpha", 1e-6, 1e-3),
        'lambda_l2': trial.suggest_float("reg_lambda", 10, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        #'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
        #'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10),
        'random_state': random_state,
        'early_stopping_rounds': 150,
        #'scale_pos_weight': scale_pos_weight,
        #'device_type': 'GPU',
    }

    # Cross-Val
    scores = []

    for fold, (tr, val) in enumerate(rkf.split(X, y)):
        X_train_, X_test_, y_train_, y_test_ = X.iloc[tr, :], X.iloc[val, :], y.iloc[tr], y.iloc[val]
        
        dtrain = lgb.Dataset(X_train_, label=y_train_)
        dvalid = lgb.Dataset(X_test_, label=y_test_)
        
        lgb_model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            #feval=rmse_metric_lgb,
            categorical_feature=CATS,
            callbacks = [
                        lgb.log_evaluation(100)]
        )

        preds = lgb_model.predict(X_test_, num_iteration=lgb_model.best_iteration)
        score = roc_auc_score(y_test_, preds)
        scores.append(score)
        
        #
        del dtrain, dvalid, X_train_, y_train_, X_test_, y_test_, lgb_model
        gc.collect()
    


    return np.mean(scores)

if SEARCH:
    
    start = time.time()
    # Optuna study for LightGBM
    study = optuna.create_study(study_name='LightGBM_optimization', direction='maximize')
    with tqdm(total=n_trials, desc="Optimizing", unit="trial") as pbar:
        study.optimize(lambda trial: objective_lgbm(trial, X_oe_concated, df_y), n_trials=n_trials, callbacks=[tqdm_callback], timeout=timeout)

    print(f'Time spent[s]: {(time.time()-start)/60:.2f} minutes')
    
    #  Results 10 trials no acelerator
    print('N trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial 

    print('  Valor: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        
    # Solo actualizamos los parámetros
    optuna_params = trial.params
    lgb_params.update(optuna_params)


#%%
# FOR LGIGHTGBM
def cross_val_lgm(X, y, kf, test, model_name="Model"):

    start= time.time()

    folds = kf # definir antes
    test_preds = np.zeros(len(test))
    oof_preds = np.zeros(len(y))
    y_vals = np.zeros(len(y))
    scores=[]

    for n_fold, (tr, val) in enumerate(rkf.split(X, y)):
        X_train_, X_test_, y_train_, y_test_ = X.iloc[tr, :], X.iloc[val, :], y.iloc[tr], y.iloc[val]
        
        dtrain = lgb.Dataset(X_train_, label=y_train_)
        dvalid = lgb.Dataset(X_test_, label=y_test_)
        
        current_time = datetime.now(pytz.utc).astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f'{model_name} - Fitting Fold {n_fold + 1} started at {current_time}')
        
        log_evaluation = lgb.log_evaluation(period=200)

        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            #feval=rmse_metric_lgb,
            categorical_feature=CATS,
            callbacks = [log_evaluation]
        )

        y_pred_val = lgb_model.predict(X_test_, num_iteration=lgb_model.best_iteration)
        oof_preds[val] = y_pred_val
        y_vals[val] = y_pred_val
        
        score = roc_auc_score(y_test_, y_pred_val)
        scores.append(score)

        test_preds += lgb_model.predict(test)/ folds.get_n_splits()

        print(f'{model_name} - Fold {n_fold + 1} ROC_AUC score : {score:.5f}')
        print('-' * 50)
        
        del dtrain, dvalid, X_train_, y_train_, X_test_, y_test_, lgb_model
        gc.collect()

    # Salida de resultados    
    print(f'{model_name} - Mean ROC_AUC score +/- std. dev.: '
        f'{np.array(scores).mean():.5f} +/- {np.array(scores).std():.3f}')
    print(f'\n{model_name} - Time spent[s]: {(time.time()-start)/60:.2f} minutes')


    return oof_preds, y_vals, test_preds, scores



#%%

preds_train, preds_test, cv_summary = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#%%

preds_train['Lightgbm'], _, preds_test['Lightgbm'], cv_summary['Lightgbm'] = \
    cross_val_lgm(X_oe_concated, df_y, rkf, test_oe_concated, model_name= 'Lightgbm')

# %%

df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['loan_status'] = preds_test['Lightgbm'].values

df_sub.to_csv('submissions/cat_34.csv', index=False)
df_sub.to_csv('ensembling2/34_pred.csv')
df_check = pd.read_csv('submissions/cat_34.csv')
display(df_check)

df_oof['loan_status'] = preds_train['Lightgbm'].values
display(df_oof)
df_oof.to_csv('ensembling2/34_oof.csv')

#%%

# %%
