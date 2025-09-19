#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Lasso ,Ridge,RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared


#%%

load_data = ['31','34', '35', '38']

oof = pd.DataFrame()
for name in load_data:
    oof[f'{name}'] = pd.read_csv(f'ensembling2/{name}_oof.csv')['loan_status'].values

preds = pd.DataFrame()
for name in load_data:
    preds[f'{name}'] = pd.read_csv(f'ensembling2/{name}_pred.csv')['loan_status'].values

display(oof)
display(preds)

# Load train
df_X = pd.read_csv('data/raw/train.csv')
df_X = df_X.drop(columns=['id'])
df_X_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_X = pd.concat([df_X, df_X_org], axis=0).reset_index(drop=True)


X_cols = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length']

df_X = df_X.loc[~df_X.duplicated(X_cols),:].reset_index(drop=True)
df_oof = pd.DataFrame(index=df_X.index)

df_y = df_X.pop('loan_status')
# Load test
df_test = pd.read_csv('data/raw/test.csv')


# %%
# CV blend oof
# Blend full data
# Pred data

df_X_oof = oof.copy()
df_y_oof = df_y.copy()

FOLDS = 15    
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=17)

best_score_list = []
for i, (train_idx, val_idx) in enumerate(kf.split(df_X_oof, df_y_oof)):
    print('#'*25)
    print(f'FOLD{i} \n')
    # SET UP DATA SPLITS
    Xtrain_CV = df_X_oof.iloc[train_idx].copy()
    ytrain_CV = df_y_oof.iloc[train_idx].copy()
    
    Xvalid_CV = df_X_oof.iloc[val_idx].copy()
    yvalid_CV = df_y_oof.iloc[val_idx].copy()

    #Blend with logistic regression
    l2mod = Ridge() #LogisticRegression()
    # Internal conversion of pandas is weird, change to numpy 
    l2mod.fit(Xtrain_CV, ytrain_CV.to_numpy().ravel())
    ypred = l2mod.predict(Xvalid_CV) #[:,1] #_proba(Xvalid_CV)[:,1]

    # CALC METRICS
    valid_roc = roc_auc_score(yvalid_CV, ypred)
    print(f"FOLD ROC: {valid_roc}")
    best_score_list.append(valid_roc)


print('#'*25)    
print(f'\n MEAN ROC: {np.mean(best_score_list)}')
print('#'*25)

#%% Blend with logistic regression
l2mod = Ridge() #LogisticRegression()
# Internal conversion of pandas is weird, change to numpy 
l2mod.fit(oof, df_y.to_numpy().ravel())
ypred = l2mod.predict(preds) #_proba(preds)
# Check order of classes
#print(l2mod.classes_)
print(ypred)

# Make submission file
df_sub = df_test[['id']].copy()
df_sub['loan_status'] = ypred #[:,1]


# %%

# SAVE PREDICTION
VER = '36'
df_sub.to_csv(f"submissions/stack_{VER}.csv",index=False)
df_sub.loc[df_sub.loan_status >= 1.0,'loan_status'] = 1.0

df_check = pd.read_csv(f'submissions/stack_{VER}.csv')
display(df_check)
print(df_check.shape)

# %%

print(len(df_check.loc[df_check.loan_status > 1]))



# %%
