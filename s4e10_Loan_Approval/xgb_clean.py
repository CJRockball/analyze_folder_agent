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
NO_BINS = 25
add_col = []

for col in ORD:
    print(f'{col} has {len(df_train[col].unique())} categories, min: {df_train[col].min()}, max: {df_train[col].max()}')
    train_max = df_train[col].max()
    train_min = df_train[col].min() 
    bins = np.linspace(train_min, train_max, NO_BINS)   
    bins[0] = -np.inf
    bins[-1] = np.inf
    labels = range(9)
    fname = f'{col}_bucket'
    #df_train[fname], bins = pd.qcut(df_train[col], 10, duplicates='drop', retbins=True) #, labels=labels)
    df_train[fname] = pd.cut(df_train[col], bins)
    df_test[fname] = pd.cut(df_test[col], bins)
    add_col.append(fname)

df_train = df_train.drop(columns=ORD)
df_test = df_test.drop(columns=ORD)
CATS = CATS + add_col
ORD = []




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
    "metric": 'binary_logloss',
    #'is_unbalance': True,
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False
}


def make_params():
    param_test ={'learning_rate': 0.02, #choices([0.1, 0.05, 0.01, 0.005, 0.001],k=1)[0],
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

xgb_params = {  "objective"             : "binary:logistic",
                      "eval_metric"           : "auc", 
                      #'device'                : "cuda" if CFG.gpu_switch == "ON" else "cpu",
                      'learning_rate'         : 0.03, 
                      #'n_estimators'          : 5_000,
                      'max_depth'             : 7, 
                      'colsample_bytree'      : 0.75, 
                      'colsample_bynode'      : 0.85,
                      'colsample_bylevel'     : 0.45,                     
                      'reg_alpha'             : 0.001, 
                      'reg_lambda'            : 0.25,
                      'verbose'               : 0,
                      'random_state'          : 42,
                      'enable_categorical'    : True,
                      #'callbacks'             : [XGBLogging(epoch_log_interval= 0)],
                      #'early_stopping_rounds' : 100, 
                     }

#%%

df_X = train_ord.copy().astype(str)
df_y = y.copy()
df_test = test_ord.copy().astype(str)

# Add noise vector to df_X
#df_X['noise'] = df_X['loan_percent_income'].sample(frac=1).values

NUM_ROUNDS = 5000
FOLDS = 5      
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# MODELS
# LGBM
xgb_model = XGBClassifier(n_estimators=5000, early_stopping_rounds=25, eval_metric=['auc'], max_bin = 262143,
                   n_jobs=4, random_state=42, colsample_bytree=0.7, max_delta_step = 0.5, gamma = 0.001, 
                      max_depth = 6, reg_alpha=0.1, reg_lambda=0.25)

# Catboost model
cat_model = CatBoostClassifier(**cat_params) #iterations=5000, loss_function='CrossEntropy', 
                        #bootstrap_type="Bayesian", random_seed=42) 


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
    
    # TARGET ENCODE
    # te = TargetEncoder(random_state=0)
    # Xtrain_CV_te = te.fit_transform(Xtrain_CV, ytrain_CV)
    # Xvalid_CV_te = te.transform(Xvalid_CV)
    # test_CV_te = te.transform(test_CV) 
    
    # TRAIN MODEL, PREDICT Y
    # xgb
    #clf = clone(xgb_model).fit(Xtrain_CV_te, ytrain_CV, eval_set=[(Xvalid_CV_te, yvalid_CV)],verbose=0)
    # # catboost
    clf = clone(cat_model).fit(Xtrain_CV, ytrain_CV, eval_set=(Xvalid_CV, yvalid_CV),) 

    # Predict gives the probability
    ypred = clf.predict_proba(Xvalid_CV)[:,1]
    if i == 0:
        pred = clf.predict_proba(test_CV)[:,1]
    else:
        pred += clf.predict_proba(test_CV)[:,1]
    # ---------------------------------------------------

    # # LIGHTGBM
#     # # PREP LGBM MODEL
#     trn_data = lgb.Dataset(Xtrain_CV, ytrain_CV)
#     lgb_eval = lgb.Dataset(Xvalid_CV, yvalid_CV)
    
#     # TRAIN MODEL, PREDICT Y
#     clf = lgb.train(params | new_params, trn_data, NUM_ROUNDS,
#                     valid_sets=lgb_eval, 
#                     callbacks=[
#                         lgb.log_evaluation(200),
#                         lgb.early_stopping(stopping_rounds=50)
#                     ])

#     # Predict gives the probability
#     ypred = clf.predict(Xvalid_CV)
#     if i == 0:
#  #       pred = clf.predict(test_CV)
#         feat_imp_split = np.array(clf.feature_importance(importance_type='split'))
#         feat_imp_gain = np.array(clf.feature_importance(importance_type='gain'))
#     else:
# #        pred += clf.predict(test_CV)
#         feat_imp_split = np.vstack((feat_imp_split,np.array(clf.feature_importance(importance_type='split'))))
#         feat_imp_gain  = np.vstack((feat_imp_gain,np.array(clf.feature_importance(importance_type='gain'))))
                    
#     # FEATURE IMPORTANCE
#     fname = clf.feature_name()
#     feat_imp_split_df = pd.DataFrame(data=feat_imp_split.T, index=fname).astype(np.int64)
#     feat_imp_split_df['split_mean'] = feat_imp_split_df.mean(axis=1)
#     feat_imp_gain_df = pd.DataFrame(data=feat_imp_gain.T, index=fname).astype(np.int64)
#     feat_imp_gain_df['gain_mean'] = feat_imp_gain_df.mean(axis=1)
#     feat_imp = pd.concat([feat_imp_split_df['split_mean'], feat_imp_gain_df['gain_mean']], axis=1)        
#     # # ------------------------------------------
    

    # CALC METRICS
    valid_roc = roc_auc_score(yvalid_CV, ypred)
    best_score_list.append(valid_roc)
    
    # INFER OOF AND TEST
    oof[val_idx] = ypred

       
# FINALIZE
#pred /= FOLDS
best_roc = np.mean(best_score_list)

print('#'*25)
print(f'Average score: {best_roc}')

#%% SAVE PREDICTION FOR SUBMISSION AND ENSEMBLING
# SAVE PREDICTION
VER = '
model_name = 'lgbm'
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

# %%

X_corr = df_X.corr()
features = X_corr.columns
len_feat = len(features)


fig, ax = plt.subplots()
im = ax.imshow(X_corr.values)
fig.colorbar(im)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len_feat), labels=features)
ax.set_yticks(np.arange(len_feat), labels=features)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len_feat):
    for j in range(len_feat):
        text = ax.text(j, i, round(X_corr.to_numpy()[i, j],1),
                       ha="center", va="center", color="w")
        


# %%
