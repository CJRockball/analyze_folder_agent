#%%
import time
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from line_profiler import profile
import logging
import time
import joblib 
from tqdm import tqdm
import lightgbm as lgb


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
                            roc_auc_score, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import train_test_split

import gc
import ctypes
# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)

clean_memory()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()
os.chdir('/home/patrick/Python/timeseries/weather/kaggle/s4e10_Loan_Approval')
      
#%% Get data

df_train = pd.read_csv('data/raw/train.csv')
df_train = df_train.drop(columns=['id'])
df_test = pd.read_csv('data/raw/test.csv')
df_test = df_test.drop(columns=['id'])
    
#%%

CATS = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
ORD = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length']
TARGET = ['loan_status']
CATS2, CATS3 = [], []
CROSS_FEATURES = CATS


#%% Set up OH
# Copy train and pop y
train_oh = df_train.copy(deep=True)
train_length = len(train_oh)
# Split y
y = train_oh.pop(TARGET[0])

# Copy test and combine
test_oh = df_test.copy(deep=True)
df_full = pd.concat([train_oh, test_oh], axis=0)
display(df_full.head())
print(train_oh.shape, test_oh.shape, df_full.shape)

# Get dummies and split in train, test
df_full = pd.get_dummies(df_full, prefix=CATS, drop_first=True, dtype=int)
train_oh = df_full.iloc[:train_length,:]
test_oh = df_full.iloc[train_length:,:]

# Get new CATS 
org_cols = df_train.columns
new_cols = train_oh.columns
new_CATS = list(set(new_cols) - set(org_cols))
CATS = new_CATS

#%% Make cross features

# def make_cross2(df_train, df_test):
#     new_columns = {}
#     new_columns2 = {}
#     for i, c1 in enumerate(CROSS_FEATURES[:-1]):
#         for j, c2 in enumerate(CROSS_FEATURES[i+1:]):
#             name = f"{c1}-{c2}"
#             new_columns[name] = df_train[c1] * df_train[c2]
#             new_columns2[name] = df_test[c1] * df_test[c2]
#             CATS2.append(name)
#             print(f"{i}-{i+j+1}, ", end='')
#     df_train = pd.concat([df_train, pd.DataFrame(new_columns)], axis=1)
#     df_test = pd.concat([df_test, pd.DataFrame(new_columns2)], axis=1)
#     print()
#     print(len(CATS2),"bi-grams generated")
#     return df_train, df_test

# new_columns = {}
# new_columns2 = {}
# CATS3 = []
# for i, c1 in enumerate(CATS[:-2]):
#     for j, c2 in enumerate(CATS[i+1:-1]):
#         for k, c3 in enumerate(CATS[i+j+2:]):
#             name = f"{c1}-{c2}-{c3}"
#             new_columns[name] = df_train[c1].astype("str") + "_" + df_train[c2].astype("str") + "_" + df_train[c3].astype("str")
#             new_columns2[name] = df_test[c1].astype("str") + "_" + df_test[c2].astype("str") + "_" + df_test[c3].astype("str")
#             CATS3.append(name)
#             print(f"{i}-{i+j+1}-{i+j+k+2}, ", end='')
# df_train = pd.concat([df_train, pd.DataFrame(new_columns)], axis=1)
# df_test = pd.concat([df_test, pd.DataFrame(new_columns2)], axis=1)
# print()
# print(len(CATS3),"tri-grams generated")

# df_X = train_oh.copy()
# df_y = y.copy()
# test_X = test_oh.copy()

# df_X, test_X = make_cross2(df_X, test_X)


#%%
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import random
from random import sample, choices

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": 'auc',
    #'is_unbalance': True,
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False
}


def make_params():
    param_test ={'learning_rate': choices([0.1, 0.05, 0.01, 0.005, 0.001],k=1)[0], #0.01, #
                 #'n_estimators': choices([150, 200, 250, 300, 350, 500],k=1)[0],
                 'num_leaves': sp_randint(30, 40).rvs(), #32, # 
                'min_child_samples': sp_randint(250, 300).rvs(), #299, #
                'min_child_weight': choices([1e-2, 1e-1, 1, 1e1, 1e2,],k=1)[0], #1, #
                'subsample': sp_uniform(loc=0.5, scale=0.2).rvs(), #0.502742, # 
                'colsample_bytree': sp_uniform(loc=0.3, scale=0.15).rvs(), #0.373231, #
                'reg_alpha': choices([1, 2, 5, 7, 10],k=1)[0], # 25, 50, 100],k=1)[0], #1, #
                'reg_lambda': choices([0, 0.5, 1, 2, 5, 7, 10],k=1)[0], #, 25, 50, 100],k=1)[0] #1, #
                } 
    return param_test

def best_params():
    param_test ={'learning_rate': 0.01, 
                 #'n_estimators': choices([150, 200, 250, 300, 350, 500],k=1)[0],
                 'num_leaves': 32,
                'min_child_samples': 299, 
                'min_child_weight': 1, 
                'subsample': 0.502742,  
                'colsample_bytree': 0.373231, 
                'reg_alpha': 1, 
                'reg_lambda': 1, 
                } 
    return param_test

df_X = train_oh.copy()
df_y = y.copy()
test_X = test_oh.copy()

NUM_ROUNDS = 5000
FOLDS = 5

#kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

def ffold_lgbm(df_X, df_y, test_X):
    new_params = best_params()
    oof = np.zeros(len(df_X))
    best_score_list = []
    for i, (train_idx, val_idx) in enumerate(kf.split(df_X, df_y)):
        print('#'*25)
        print(f' FOLD {i} \n')
        # SET UP DATA SPLITS
        Xtrain_CV = df_X.iloc[train_idx].copy()
        ytrain_CV = df_y.iloc[train_idx].copy()
        
        Xvalid_CV = df_X.iloc[val_idx].copy()
        yvalid_CV = df_y.iloc[val_idx].copy()

        test_CV = test_X.copy()

        # PREP LGBM MODEL
        trn_data = lgb.Dataset(Xtrain_CV, ytrain_CV)
        lgb_eval = lgb.Dataset(Xvalid_CV, yvalid_CV)
        
        # TRAIN MODEL, PREDICT Y
        clf = lgb.train(params | new_params, trn_data, NUM_ROUNDS,
                        valid_sets=lgb_eval, 
                        callbacks=[
                            lgb.log_evaluation(100),
                            lgb.early_stopping(stopping_rounds=100)
                        ])
            
        # Predict gives the probability
        ypred = clf.predict(Xvalid_CV)
        
        # CALC METRICS
        valid_roc = roc_auc_score(yvalid_CV, ypred)
        best_score_list.append(valid_roc)
        
        # INFER OOF AND TEST
        oof[val_idx] = ypred
        if i == 0:
            pred = clf.predict(test_CV)
            feat_imp_split = np.array(clf.feature_importance(importance_type='split'))
            feat_imp_gain = np.array(clf.feature_importance(importance_type='gain'))
        else:
            pred += clf.predict(test_CV)
            feat_imp_split = np.vstack((feat_imp_split,np.array(clf.feature_importance(importance_type='split'))))
            feat_imp_gain  = np.vstack((feat_imp_gain,np.array(clf.feature_importance(importance_type='gain'))))
            
        
        # FEATURE IMPORTANCE
        fname = clf.feature_name()
        feat_imp_split_df = pd.DataFrame(data=feat_imp_split.T, index=fname).astype(np.int64)
        feat_imp_split_df['split_mean'] = feat_imp_split_df.mean(axis=1)
        feat_imp_gain_df = pd.DataFrame(data=feat_imp_gain.T, index=fname).astype(np.int64)
        feat_imp_gain_df['gain_mean'] = feat_imp_gain_df.mean(axis=1)
        feat_imp = pd.concat([feat_imp_split_df['split_mean'], feat_imp_gain_df['gain_mean']], axis=1)
        
    # FINALIZE
    pred /= FOLDS
    best_roc = np.mean(best_score_list)
    
    print('#'*25)
    print(best_score_list)
    print(f'Average score: {best_roc}')
    print('#'*25, '\n')
        
    return oof, pred, clf, new_params, feat_imp

oof, pred, clf, new_params, feat_imp = ffold_lgbm(df_X, df_y, test_X)

#%% SAVE PREDICTION FOR SUBMISSION AND ENSEMBLING
# SAVE PREDICTION
VER = '1'
sub = pd.read_csv("data/raw/sample_submission.csv")
sub.loan_status = pred
sub.to_csv(f"ensembling/lgbm_pred_{VER}.csv",index=False)

df_check = pd.read_csv(f'ensembling/lgbm_pred_{VER}.csv')
display(df_check)
print(df_check.shape)

# SAVE OOF
df_get_id = pd.read_csv('data/raw/train.csv')
oof_df = df_get_id[["id"]].copy()
oof_df[f"pred_{VER}"] = oof
oof_df.to_csv(f"ensembling/lgbm_oof_{VER}.csv",index=False)

df_check = pd.read_csv(f'ensembling/lgbm_oof_{VER}.csv')
display(df_check)
print(df_check.shape)

#%% PLOT FEATURE IMPORTANCE
# Plot split mean importance
feat_sort1 = feat_imp.sort_values(by=['split_mean'], ascending=False).iloc[:20,:]
plt.show()
plt.barh(feat_sort1.index, feat_sort1.split_mean)
plt.title('Feature Importance Split')
plt.show()
# Plot gain importance
feat_sort2 = feat_imp.sort_values(by=['gain_mean'], ascending=False).iloc[:20,:]
plt.show()
plt.barh(feat_sort2.index, feat_sort2.gain_mean)
plt.title('Feature Importance Gain')
plt.show()

# Print split, gain x most important features
print(feat_sort1.index)
print(feat_sort2.index)

#%% Extract most important features

feature_list = feat_sort2.index
print(feature_list)

# Remove single features
add_feature = [x for x in feature_list if '-' in x ]

print(len(add_feature))
print(add_feature)

feature_list = add_feature + CATS + ORD

# %%

df_X_feat_sel = df_X.copy()
test_X_feat_sel = test_X.copy()
df_X_feat_sel = df_X_feat_sel[feature_list]
test_X_feat_sel = test_X_feat_sel[feature_list]

oof, pred, clf, new_params, feat_imp = ffold_lgbm(df_X_feat_sel, df_y, test_X_feat_sel)


# %%
