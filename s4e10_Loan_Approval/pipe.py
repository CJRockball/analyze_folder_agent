#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso
from sklearn.base import clone

from sklearn.ensemble import  RandomForestClassifier, HistGradientBoostingClassifier,\
    GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier

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

# df_train = df_train.drop(df_train[df_train['person_age'] > 100].index)
# df_train = df_train.drop(df_train[df_train['person_emp_length'] > 100].index)
# df_train = df_train.reset_index(drop=True)

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
y = train_ord.pop(TARGET[0])

#%%


df_X = train_ord.copy()
df_y = y.copy()
df_test = test_ord.copy()

FOLDS = 5
            
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(df_X))
best_score_list = []
for i, (train_idx, val_idx) in enumerate(kf.split(df_X, df_y)):
    
    # SET UP DATA SPLITS
    Xtrain_CV = df_X.iloc[train_idx].copy()
    ytrain_CV = df_y.iloc[train_idx].copy()
    
    Xvalid_CV = df_X.iloc[val_idx].copy()
    yvalid_CV = df_y.iloc[val_idx].copy()

    test_CV = df_test.copy()
    
    # TRAIN MODEL, PREDICT Y
    # xgb
    clf = AdaBoostClassifier() #ExtraTreesClassifier() # GradientBoostingClassifier() #HistGradientBoostingClassifier() # RandomForestClassifier()# AdaBoostClassifier() #
    clf.fit(Xtrain_CV, ytrain_CV)

    # Predict gives the probability
    ypred = clf.predict_proba(Xvalid_CV)[:,1]

    # CALC METRICS
    valid_roc = roc_auc_score(yvalid_CV, ypred)
    print(f'Fold {i}, ROC: {valid_roc}')
    best_score_list.append(valid_roc)
    
    # INFER OOF AND TEST
    oof[val_idx] = ypred
    if i == 0:
        pred = clf.predict_proba(test_CV)[:,1]
    else:
        pred += clf.predict_proba(test_CV)[:,1]

       
# FINALIZE
pred /= FOLDS
best_roc = np.mean(best_score_list)

print('#'*25)
print(f'\nFold ROC: {best_score_list}')
print(f'Average score: {best_roc}')

#%% SAVE PREDICTION FOR SUBMISSION AND ENSEMBLING
# SAVE PREDICTION
VER = 'basic'
model_name = 'ada'
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





# %%
