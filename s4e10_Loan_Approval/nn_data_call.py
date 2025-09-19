#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import AUROC, BinaryAUROC
from nn_cv_fcn import nn_algo


#%%

df_test = pd.read_csv('data/raw/test.csv').assign(source=0)
df_train = pd.read_csv('data/raw/train.csv').assign(source=0) #.drop(columns=['id'])
df_org = pd.read_csv('data/raw/credit_risk_dataset.csv').assign(source=1)
TARGET = ['loan_status']

df_comb = pd.concat([df_train, df_org], axis=0).reset_index(drop=True)



def to_rank(col):
       # Dicretize from 0 to N
       return col.fillna(-1).rank(method='dense').astype('int') - 1

def fe(df):
       cat_cols = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

       # Discretize all num features:
       df['cb_person_cred_hist_length'] = to_rank(df['cb_person_cred_hist_length'])
       df['loan_amnt'] = to_rank(df['loan_amnt'])
       df['person_income'] = to_rank(df['person_income'])
       df['loan_int_rate'] = to_rank(df['loan_int_rate'])
       df['person_emp_length'] = to_rank(df['person_emp_length'])
       df['loan_percent_income'] = to_rank(df['loan_percent_income'])
       df['person_age'] = to_rank(df['person_age'])       

       for col in cat_cols:
              # Count + rank encoding less to more frequent
              col_series = df[col].fillna('#NA#')
              # Get all groups from cat feature
              mapping = col_series.value_counts().to_dict()
              # Start the mapping at 0
              code_as = 0
              # Build up converter dict
              for i,key in enumerate(reversed(mapping)):
                     mapping[key] = code_as
                     code_as += 1
              # Change all group names in features
              df[col] = col_series.map(mapping)
              df[col] = df[col].astype('int')
       return df

# Discretize all features, fillna, and map CATS to ordinal
df_all = fe(pd.concat([df_comb, df_test]))

# Sort out train, val and test
idxs = (~df_all[TARGET[0]].isna()) & (df_all['source'] == 0)
train_data = df_all[idxs].reset_index(drop=True)
idxs = (df_all[TARGET[0]].isna()) & (df_all['source'] == 0)
test_data = df_all[idxs].drop(columns=[TARGET[0]])
org_data = df_all.query('source == 1')

#%%

df_train = pd.concat([train_data, org_data]).reset_index(drop=True)
df_train = df_train.drop(columns=['source', 'id'])
df_test = test_data.drop(columns=['source','id'])
y = df_train.pop(TARGET[0])

NUMS = ['cb_person_default_on_file', ]
CATS = [
       'person_home_ownership',
       'loan_intent',
       'loan_grade',
       'person_emp_length',
       'loan_int_rate',
       'loan_percent_income',
       'person_age',
       'person_income',
       'loan_amnt',
       'cb_person_cred_hist_length']

#%%

nn_algo(df_train, y, df_test, NUMS, CATS)


# %%
