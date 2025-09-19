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

from catboost import CatBoostClassifier, Pool

from sklearn.preprocessing import TargetEncoder
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

df_X = pd.read_csv('data/raw/train.csv')
df_X = df_X.drop(columns=['id'])
df_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_X = pd.concat([df_X, df_org], axis=0).reset_index(drop=True)

X_cols = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length']

df_X = df_X.loc[~df_X.duplicated(X_cols),:].reset_index(drop=True)
df_oof = pd.DataFrame(index=df_X.index)
df_y = df_X.pop('loan_status')


df_test = pd.read_csv('data/raw/test.csv')
df_test = df_test.drop(columns=['id'])

# df_X['loan_percent_income_diff'] = df_X['loan_amnt'] / df_X['person_income'] - df_X['loan_percent_income']
# df_test['loan_percent_income_diff'] = df_test['loan_amnt'] / df_test['person_income'] - df_test['loan_percent_income']

#%%
# Determine the numerical variables
TARGET = ['loan_status']
NUM = [col for col in df_X.columns if df_X[col].dtype in ['int64', 'float64']]
# Determine the categorical variables
CATS = [col for col in df_X.columns if df_X[col].dtype == 'object']

# cat indices
cat_indices = [df_X.columns.get_loc(col) for col in CATS]

#%%
# For catboost we consider all columns as string
cat_list = df_X.columns.values

df_cat = df_X.copy()
df_cat = df_cat.fillna(0)

df_cat = df_cat.astype('string')  

#%% Make all col string for catboos
df_X = df_X.astype(str)
df_test = df_test.astype(str)

#%% Cat params
cat_params={
    'iterations': 1500,
    'depth': 6,
    'eta': 0.28901888228959255, 
    'reg_lambda': 41.0642500499563, 
    #'colsample_bylevel': 0.6,
    #'subsample': 0.8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'cat_features': cat_list, 
    'random_state': random_state,
    'min_data_in_leaf': 51,
    'early_stopping_rounds': 150,
    #'max_bin': 5000, # Can try to blow out max_bins. Didn't work here
    'verbose':200,
    #'random_strength': 1.5,
    #'bootstrap_type': 'Bernoulli',
}

#%% Set up optuna search

rkf = RepeatedKFold(n_splits=15, n_repeats=1, random_state=random_state)
SEARCH= False
n_trials = 100
timeout= 3600*3

#%% Optuna
# Define el objetivo para Optuna
def objective_cat(trial, X, y):
    iterations = 1500
    params = {
    'depth': trial.suggest_int('depth', 4, 10),
    'eta': trial.suggest_float('eta', 0.1, 0.3),
    'reg_lambda': trial.suggest_float("reg_lambda", 1, 50),
    #'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.5, 0.8),
    #'subsample': 0.8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'cat_features': cat_list,
    'random_state': random_state,
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
    'early_stopping_rounds': 150,
    'verbose':0,
    #'random_strength': 1.5,
    #'bootstrap_type': 'Bernoulli',
    #'task_type': 'GPU'
    }

    # Cross-Val
    scores = []

    for fold, (tr, val) in enumerate(rkf.split(X, y)):
        X_train_, X_test_, y_train_, y_test_ = X.iloc[tr, :], X.iloc[val, :], y.iloc[tr], y.iloc[val]
        
        train_pool = Pool(data=X_train_, label=y_train_, cat_features=cat_list)
        valid_pool = Pool(data=X_test_, label=y_test_, cat_features=cat_list)
        
        cat_model = CatBoostClassifier(**cat_params,)
        
        cat_model.fit(train_pool, eval_set=valid_pool, verbose=0)
        
        # Usar la mejor iteración para la predicción
        best_iter = cat_model.best_iteration_
            
        preds = cat_model.predict_proba(X_test_, ntree_end=best_iter)[:, 1]        
        score = roc_auc_score(y_test_, preds)
        scores.append(score)
        
        del X_train_, y_train_, X_test_, y_test_, cat_model
        gc.collect()
    
    return np.mean(scores)

if SEARCH:
   
    start = time.time()
    # Optuna study for Catboost
    study = optuna.create_study(study_name='Catboost_optimization', direction='maximize')
    with tqdm(total=n_trials, desc="Optimizing", unit="trial") as pbar:
        study.optimize(lambda trial: objective_cat(trial, df_X, df_y), n_trials=n_trials, callbacks=[tqdm_callback], timeout=timeout)

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
    cat_params.update(optuna_params)


#%%

def cross_val_cat(X, y, kf, test, model_name="Model"):

    start= time.time()

    folds = kf # definir antes
    test_preds = np.zeros(len(test))
    oof_preds = np.zeros(len(y))
    y_vals = np.zeros(len(y))
    scores=[]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
        X_train_, y_train_ = X.iloc[train_idx], y.iloc[train_idx]
        X_val_, y_val_ = X.iloc[valid_idx], y.iloc[valid_idx]

        current_time = datetime.now(pytz.utc).astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f'{model_name} - Fitting Fold {n_fold + 1} started at {current_time}')
        
        train_pool = Pool(data=X_train_, label=y_train_, cat_features=cat_list)
        valid_pool = Pool(data=X_val_, label=y_val_, cat_features=cat_list)
        
        cat_model = CatBoostClassifier(**cat_params,)
        
        cat_model.fit(train_pool, eval_set=valid_pool, verbose=200)
        
        # Usar la mejor iteración para la predicción
        best_iter = cat_model.best_iteration_
            
        y_pred_val = cat_model.predict_proba(X_val_, ntree_end=best_iter)[:, 1] 
        oof_preds[valid_idx] = y_pred_val
        y_vals[valid_idx] = y_val_
        
        score = roc_auc_score(y_val_, y_pred_val)
        scores.append(score)
        
        test_preds += cat_model.predict_proba(test, ntree_end=best_iter)[:, 1]/ folds.get_n_splits()

        print(f'{model_name} - Fold {n_fold + 1} RMSE score : {score:.5f}')
        print('-' * 50)
        
        del X_train_, y_train_, X_val_, y_val_, cat_model
        gc.collect()

    # Salida de resultados    
    print(f'{model_name} - Mean RMSE score +/- std. dev.: '
        f'{np.array(scores).mean():.5f} +/- {np.array(scores).std():.3f}')
    print(f'\n{model_name} - Time spent[s]: {(time.time()-start)/60:.2f} minutes')

    return oof_preds, y_vals, test_preds, scores

#%%

preds_train, preds_test, cv_summary = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#%%

preds_train['Catboost'], _, preds_test['Catboost'], cv_summary['Catboost'] = \
            cross_val_cat(df_X, df_y, rkf, df_test, model_name= 'Catboost')

# %%

df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['loan_status'] = preds_test['Catboost'].values

df_sub.to_csv('submissions/cat_31.csv', index=False)
df_sub.to_csv('ensembling2/31_pred.csv')
df_check = pd.read_csv('submissions/cat_31.csv')
display(df_check)

df_oof['loan_status'] = preds_train['Catboost'].values
display(df_oof)
df_oof.to_csv('ensembling2/31_oof.csv')



# %%
