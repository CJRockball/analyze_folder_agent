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
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
    
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

# Remove duplicates
df_X = df_X.loc[~df_X.duplicated(X_cols),:].reset_index(drop=True)
# Fix nulls
has_nulls = ['person_emp_length', 'loan_int_rate']
df_X[has_nulls] = df_X[has_nulls].fillna(df_X[has_nulls].mean())
# Remove outliers
df_X = df_X.loc[df_X['person_age'] < 100 ]
df_X = df_X.loc[df_X['person_emp_length'] < 100 ]
df_X = df_X.loc[df_X['person_income'] < 3e6 ]

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

display(df_X)

df_X.boxplot(['person_age']) #, by=['X', 'Y'])
plt.show()

#%% Categorical data processing
# order = [['OWN','MORTGAGE','RENT', 'OTHER',], 
#          ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'],
#          ['A', 'B', 'C', 'F','D', 'E', 'G'], ['N','Y']]
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_oe = pd.DataFrame(oe.fit_transform(df_X[CATS]), columns=CATS).fillna(-1).astype(int)
test_oe = pd.DataFrame(oe.transform(df_test[CATS]), columns=CATS).fillna(-1).astype(int)

X_oe_c = pd.concat([X_oe, df_X[NUMS]], axis= 1)
test_oe_c = pd.concat([test_oe, df_test[NUMS]], axis= 1)

print(len(X_oe))
print(len(df_X[NUMS]))
print(len(X_oe_c))

print(len(test_oe))
print(len(df_test))
print(len(test_oe_c))

#%% Chek

for i, cname in enumerate(oe.get_feature_names_out()):
    print(cname)
    print({v:i for i,v in enumerate(oe.categories_[i])})


#%%

import itertools

perm_list = list(itertools.permutations([0,1,2,3]))

df_result = pd.DataFrame(columns=['order', 'auc_train', 'auc_test', 'auc_oob'])
#mapping = {3:3, 2:0, 1:1, 0:2}

perm_list = [(0,1,2,3)]
# OTHER, RENT, OWN, MORTGAGE
for i,perm_tuple in enumerate(perm_list):
    mapping = {i:v for i,v in enumerate(perm_tuple)}
    X_oe_c['person_home_ownership'] = X_oe_c['person_home_ownership'].replace(mapping)
    # mapping = {0:0, 1:2, 2:3, 3:1, 4:4,5:5}
    # X_oe_c['loan_intent'] = X_oe_c['loan_intent'].replace(mapping)


    #

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_oe_c, df_y, test_size=0.3, random_state=42)

    #

    model = RandomForestClassifier(min_samples_leaf=75, bootstrap=True,
                                oob_score=roc_auc_score,
                                random_state=42,
                                n_jobs=4) #max_leaf_nodes=4)
    model.fit(Xtrain,ytrain)

    # Check fit
    from sklearn.metrics import roc_auc_score


    y_pred_train = model.predict_proba(Xtrain)[:, 1]
    y_pred_test = model.predict_proba(Xtest)[:, 1]

    auc_train = roc_auc_score(ytrain, y_pred_train)
    auc_test  = roc_auc_score(ytest, y_pred_test)

    print(perm_tuple)
    # print(f'train auc score: {auc_train}')
    # print(f'test auc score: {auc_test}')
    # print(f'Model oob auc score {model.oob_score_}')
    
    new_row = {'order':str(perm_tuple), 
               'auc_train':auc_train, 
               'auc_test': auc_test, 
               'auc_oob':model.oob_score_}
    df_result.loc[i] = new_row


#%%


display(df_result.sort_values(by=['auc_test']))


#%%
from sklearn import tree
from sklearn.tree import export_graphviz
import re
import graphviz
import IPython

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    
    s=export_graphviz(t, out_file=None, feature_names=df.columns,
                      filled=True,special_characters=True, rotate=False,
                      precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',f'Tree {{ size={size}; ratio={ratio}', s)))


draw_tree(model.estimators_[0], Xtrain, size=8, precision=2)


# %%
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
    
fi = rf_feat_importance(model, Xtrain)   
print(fi)  
    
fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plt.show()

# %% Check correlation with rank-correlation
# Pearson, SPearman, Mutual Information, PhiK
no_features = len(Xtrain.columns)
p_corr = Xtrain.corr('spearman')

fig, ax = plt.subplots()
im = ax.imshow(p_corr.values)
fig.colorbar(im)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(no_features), labels=Xtrain.columns)
ax.set_yticks(np.arange(no_features), labels=Xtrain.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(no_features):
    for j in range(no_features):
        text = ax.text(j, i, round(p_corr.to_numpy()[i, j],2),
                       ha="center", va="center", color="w")
        
    
# %% Remove one feature at the time and check the auc_roc

def get_oob(df,y):
    m = RandomForestClassifier(min_samples_leaf=75, bootstrap=True,
                               oob_score=roc_auc_score,
                               n_jobs=4)
    m.fit(df, y)
    return m.oob_score_

# Baseline
get_oob(Xtrain,ytrain)
# One fature at the time
check_feat = {c:get_oob(Xtrain.drop(c, axis=1),ytrain) for c in X_cols}
# Print
for k,v in check_feat.items():
    print(k, v)

# %% Check partial dependence
from sklearn.inspection import PartialDependenceDisplay

# fig,ax = plt.subplots(figsize=(12, 4))
# plot_partial_dependence(model, Xtrain, ['loan_percent_income','loan_grade'],
#                         grid_resolution=20, ax=ax);
features = [3] #, 1, (0, 1)]
PartialDependenceDisplay.from_estimator(model, Xtrain, features, categorical_features=np.array([0,1,2,3])) #, kind='both', centered=True)
# ax = plt.gca()
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['OTHER', 'RENT', 'OWN', 'MORTGAGE'])


# %%

print(Xtrain.columns)


#%%

df_plot = pd.concat([df_X, df_y], axis=1)

df_plot.loc[df_plot['loan_status'] == 1, 'loan_int_rate'].hist()
plt.show()

df_plot.loc[df_plot['loan_status'] == 0, 'loan_int_rate'].hist()
plt.show()


# %%

display(df_X.head(5))

# %%
