#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso
from sklearn.base import clone

import xgboost as xgb
from xgboost import XGBClassifier

from catboost import CatBoostClassifier 


#%%

df_train = pd.read_csv('data/raw/train.csv')
df_train = df_train.drop(columns=['id'])
df_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_train = pd.concat([df_train, df_org], axis=0)

df_test = pd.read_csv('data/raw/test.csv')
df_test = df_test.drop(columns=['id'])

#%%

print(df_train.isnull().sum())
has_nulls = ['person_emp_length', 'loan_int_rate']
df_train[has_nulls] = df_train[has_nulls].fillna(df_train[has_nulls].mean())
print(df_train.isnull().sum())


change_list = ['loan_int_rate', 'loan_percent_income']
df_train[change_list] = (df_train[change_list] * 100).astype(int)
df_test[change_list] = (df_test[change_list] * 100).astype(int)

#%%

CATS = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
ORD = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length']
TARGET = ['loan_status']
CATS2, CATS3 = [], []

#%% NUM TRANSFORM

# def num_transform(train, test, col_list):
#     add_cols = []
#     for name in ['person_income']:
#         train[f'log_{name}'] = np.log(train[name])
#         test[f'log_{name}'] = np.log(test[name])
#         add_cols.append(f'log_{name}')

#         train[f'sqrt_{name}'] = np.sqrt(train[name])
#         test[f'sqrt_{name}'] = np.sqrt(test[name])
#         add_cols.append(f'sqrt_{name}')
        
#         train[f'sq_{name}'] = train[name] * train[name]
#         test[f'sq_{name}'] = test[name] * test[name]
#         add_cols.append(f'sq_{name}')
        
    

#     return train, test, add_cols

# mod_list = ['person_age', 'person_income', 'person_emp_length']
# df_train, df_test, add_cols = num_transform(df_train, df_test, mod_list)
# ORD = ORD + add_cols

# #%% CAT TRANSFORM

# CROSS_FEATURES = CATS

# new_columns = {}
# new_columns2 = {}
# for i, c1 in enumerate(CROSS_FEATURES[:-1]):
#     for j, c2 in enumerate(CROSS_FEATURES[i+1:]):
#         name = f"{c1}-{c2}"
#         new_columns[name] = df_train[c1].astype("str") + "_" + df_train[c2].astype("str")
#         new_columns2[name] = df_test[c1].astype("str") + "_" + df_test[c2].astype("str")
#         CATS2.append(name)
#         print(f"{i}-{i+j+1}, ", end='')
# df_train = pd.concat([df_train, pd.DataFrame(new_columns)], axis=1)
# df_test = pd.concat([df_test, pd.DataFrame(new_columns2)], axis=1)
# print()
# print(len(CATS2),"bi-grams generated")

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
y = train_ord.pop(TARGET[0])

# %% PARAMS

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import random
from random import sample, choices

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": 'logloss',
    #'is_unbalance': True,
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False
}


def make_params():
    param_test ={'learning_rate': 0.01, #choices([0.1, 0.05, 0.01, 0.005, 0.001],k=1)[0],
                 #'n_estimators': choices([150, 200, 250, 300, 350, 500],k=1)[0],
                 'num_leaves': 32, #sp_randint(30, 40).rvs(), 
                'min_child_samples': 299, #sp_randint(250, 300).rvs(), 
                'min_child_weight':1, #choices([1e-2, 1e-1, 1, 1e1, 1e2,],k=1)[0],
                'subsample': 0.502742, #sp_uniform(loc=0.5, scale=0.2).rvs(), 
                'colsample_bytree': 0.373231, #sp_uniform(loc=0.3, scale=0.15).rvs(),
                'reg_alpha':1, #choices([1, 2, 5, 7, 10],k=1)[0], # 25, 50, 100],k=1)[0],
                'reg_lambda':1 # choices([0, 0.5, 1, 2, 5, 7, 10],k=1)[0], #, 25, 50, 100],k=1)[0]
                } 
    return param_test

cat_params={
    'iterations': 5000,
    'depth': 6,
    'eta': 0.28901888228959255, 
    'reg_lambda': 41.0642500499563, 
    #'colsample_bylevel': 0.6,
    #'subsample': 0.8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    #'cat_features': cat_list, 
    'random_state': 42,
    'min_data_in_leaf': 51,
    'early_stopping_rounds': 150,
    'verbose':200,
    #'random_strength': 1.5,
    #'bootstrap_type': 'Bernoulli',
}

#%%


df_X = train_ord.copy()
df_y = y.copy()
df_test = test_ord.copy()

NUM_ROUNDS = 5000
FOLDS = 5

# Add brand mean and std
# mean_features = ['person_home_ownership', 'loan_grade']
# df_full = pd.concat([df_X, df_y], axis=1)
# for name in mean_features:
#     lpi_mean = df_full.groupby(name)['loan_status'].agg(['mean']). \
#                                 rename(columns={'mean':f'{name}_mean'}).to_dict()
#     for k in lpi_mean.keys():
#         df_test[k] = df_test[name].map(lpi_mean[k])
            

#kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
model = XGBClassifier(n_estimators=2000, early_stopping_rounds=100, eval_metric=['auc'], max_bin = 262143,
                   n_jobs=4, random_state=0, colsample_bytree=0.7, max_delta_step = 0.5, gamma = 0.001, 
                      max_depth = 6, device="cuda")

new_params = make_params()
oof = np.zeros(len(df_X))
best_score_list = []
for i, (train_idx, val_idx) in enumerate(kf.split(df_X, df_y)):
    # SET UP DATA SPLITS
    Xtrain_CV = df_X.iloc[train_idx].copy()
    ytrain_CV = df_y.iloc[train_idx].copy()
    
    Xvalid_CV = df_X.iloc[val_idx].copy()
    yvalid_CV = df_y.iloc[val_idx].copy()

    test_CV = df_test.copy()
    
    # SET UP MEAN FEATURES
    # df_ord = pd.concat([Xtrain_CV, ytrain_CV], axis=1)
    # for name in mean_features:
    #     lpi_mean = df_ord.groupby(name)['loan_status'].agg(['mean']). \
    #                                 rename(columns={'mean':f'{name}_mean'}).to_dict()
    #     for k in lpi_mean.keys():
    #         Xtrain_CV[k] = Xtrain_CV[name].map(lpi_mean[k])
    #         Xvalid_CV[k] = Xvalid_CV[name].map(lpi_mean[k])

    
    # TRAIN MODEL, PREDICT Y
    clf = clone(model).fit(Xtrain_CV, ytrain_CV, eval_set=[(Xvalid_CV, yvalid_CV)],verbose=0)

    # Catboost model
    # clf = CatBoostClassifier(**cat_params) #iterations=5000, loss_function='CrossEntropy', 
    #                         #bootstrap_type="Bayesian", random_seed=42) 
    # clf.fit(Xtrain_CV, ytrain_CV, eval_set=(Xvalid_CV, yvalid_CV), 
    #         #early_stopping_rounds=50,
    #         #verbose=100
    #         ) 
    
    # Predict gives the probability
    ypred = clf.predict_proba(Xvalid_CV)[:,1]
    
    # CALC METRICS
    valid_roc = roc_auc_score(yvalid_CV, ypred)
    best_score_list.append(valid_roc)
    
    # INFER OOF AND TEST
    oof[val_idx] = ypred
    if i == 0:
        pred = clf.predict_proba(test_CV)[:,1]
        # feat_imp_split = np.array(clf.feature_importance(importance_type='split'))
        # feat_imp_gain = np.array(clf.feature_importance(importance_type='gain'))
    else:
        pred += clf.predict_proba(test_CV)[:,1]
        # feat_imp_split = np.vstack((feat_imp_split,np.array(clf.feature_importance(importance_type='split'))))
        # feat_imp_gain  = np.vstack((feat_imp_gain,np.array(clf.feature_importance(importance_type='gain'))))
        
    
    # FEATURE IMPORTANCE
    # fname = clf.feature_name()
    # feat_imp_split_df = pd.DataFrame(data=feat_imp_split.T, index=fname).astype(np.int64)
    # feat_imp_split_df['split_mean'] = feat_imp_split_df.mean(axis=1)
    # feat_imp_gain_df = pd.DataFrame(data=feat_imp_gain.T, index=fname).astype(np.int64)
    # feat_imp_gain_df['gain_mean'] = feat_imp_gain_df.mean(axis=1)
    # feat_imp = pd.concat([feat_imp_split_df['split_mean'], feat_imp_gain_df['gain_mean']], axis=1)
       
# FINALIZE
pred /= FOLDS
best_roc = np.mean(best_score_list)

print('#'*25)
print(f'Average score: {best_roc}')

#%% SAVE PREDICTION FOR SUBMISSION AND ENSEMBLING
# SAVE PREDICTION
VER = 'basic'
model_name = 'xgb'
sub = pd.read_csv("data/raw/sample_submission.csv")
sub.loan_status = pred
sub.to_csv(f"ensembling/{model_name}_pred_{VER}.csv",index=False)

df_check = pd.read_csv(f'ensembling/{model_name}_pred_{VER}.csv')
display(df_check)
print(df_check.shape)

# SAVE OOF
#df_get_id = pd.read_csv('data/raw/train.csv')
#oof_df = df_get_id[["id"]].copy()
oof_df = pd.DataFrame()
oof_df.index = df_train.index
oof_df[f"pred_{VER}"] = oof
oof_df.to_csv(f"ensembling/{model_name}_oof_{VER}.csv",index=False)

df_check = pd.read_csv(f'ensembling/{model_name}_oof_{VER}.csv')
display(df_check)
print(df_check.shape)

#%% PLOT FEATURE IMPORTANCE

feat_sort1 = feat_imp.sort_values(by=['split_mean']).iloc[:30,:]
plt.show()
plt.barh(feat_sort1.index, feat_sort1.split_mean)
plt.title('Feature Importance Split')
plt.show()

feat_sort2 = feat_imp.sort_values(by=['gain_mean']).iloc[:30,:]
plt.show()
plt.barh(feat_sort2.index, feat_sort2.gain_mean)
plt.title('Feature Importance Gain')
plt.show()

feature_list = feat_sort2.index
print(feat_sort1.index)
print(feat_sort2.index)

# %%

print(df_X.columns)
# %%
