#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso
from sklearn.base import clone

import xgboost as xgb
from xgboost import XGBClassifier

from catboost import CatBoostClassifier 

import lightgbm as lgb

#%% Load data, concat original dataset

df_train = pd.read_csv('data/raw/train.csv')
df_train = df_train.drop(columns=['id'])
df_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_train = pd.concat([df_train, df_org], axis=0)

df_test = pd.read_csv('data/raw/test.csv')
df_test = df_test.drop(columns=['id'])

#%% Make feature type lists

CATS = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
ORD = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length']
TARGET = ['loan_status']
CATS2, CATS3 = [], []

#%% Impute nans, 

print(df_train.isnull().sum())
has_nulls = ['person_emp_length', 'loan_int_rate']
df_train[has_nulls] = df_train[has_nulls].fillna(df_train[has_nulls].mean())
print(df_train.isnull().sum())

#%% Remove outliers

df_train = df_train.drop(df_train[df_train['person_age'] > 100].index)
df_train = df_train.drop(df_train[df_train['person_emp_length'] > 100].index)
df_train = df_train.reset_index(drop=True)


#%% NUM TRANSFORM

def num_transform(train, test, col_list):
    add_cols = []
    for name in ['person_income']:
        train[f'log_{name}'] = np.log(train[name])
        test[f'log_{name}'] = np.log(test[name])
        add_cols.append(f'log_{name}')
    return train, test, add_cols

mod_list = ['person_age', 'person_income', 'person_emp_length']
df_train, df_test, add_cols = num_transform(df_train, df_test, mod_list)
ORD = ORD + add_cols

#%%
train_ord = df_train.copy(deep=True)
test_ord = df_test.copy(deep=True)
FEATURE_CATS = CATS + CATS2 + CATS3

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,)
train_ord[FEATURE_CATS] = oe.fit_transform(train_ord[FEATURE_CATS])
test_ord[FEATURE_CATS] = oe.transform(test_ord[FEATURE_CATS])

# # Put all unseen classes in class 0
train_ord[FEATURE_CATS] = train_ord[FEATURE_CATS] + 1
test_ord[FEATURE_CATS] = test_ord[FEATURE_CATS] + 1
test_ord[FEATURE_CATS] =  test_ord[FEATURE_CATS].fillna(0)
# Split y
y = train_ord.pop(TARGET[0]).to_frame()

#%%

train_ord.to_parquet('data/artifacts/train_ord.parquet')
test_ord.to_parquet('data/artifacts/test_ord.parquet')
y.to_parquet('data/artifacts/y_ord.parquet')

# %%
