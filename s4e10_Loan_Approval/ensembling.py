#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Lasso ,Ridge,RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load training level1 features
oof1 = pd.read_csv(f'ensembling/lgbm_oof_basic.csv').rename(columns={'pred_basic':'pred_1'}) #.drop(columns=['id'])
oof2 = pd.read_csv(f'ensembling/xgb_oof_basic.csv').rename(columns={'pred_basic':'pred_2'}) #.drop(columns=['id'])
oof3 = pd.read_csv(f'ensembling/cat_oof_basic.csv').rename(columns={'pred_basic':'pred_3'}) #.drop(columns=['id'])

# Set up training data
oof_train = pd.concat([oof1, oof2, oof3], axis=1)

# Load predicting data
pred1 = pd.read_csv('ensembling/lgbm_pred_basic.csv').rename(columns={'loan_status': 'pred_1'}).drop(columns=['id'])
pred2 = pd.read_csv('ensembling/xgb_pred_basic.csv').rename(columns={'loan_status': 'pred_2'}).drop(columns=['id'])
pred3 = pd.read_csv('ensembling/cat_pred_basic.csv').rename(columns={'loan_status': 'pred_3'}).drop(columns=['id'])
# Set up testing features
pred_features = pd.concat([pred1, pred2, pred3], axis=1)

# Load train
df_X = pd.read_csv('data/raw/train.csv')
df_X = df_X.drop(columns=['id'])
df_X_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_X = pd.concat([df_X, df_X_org], axis=0)
df_y = df_X[['loan_status']]
# Load test
df_test = pd.read_csv('data/raw/test.csv')

#%% Blend in CV


df_X_oof = oof_train.copy()
df_y_oof = df_y.copy()

FOLDS = 5    
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
    l2mod = LogisticRegression()
    # Internal conversion of pandas is weird, change to numpy 
    l2mod.fit(Xtrain_CV, ytrain_CV.to_numpy().ravel())
    ypred = l2mod.predict_proba(Xvalid_CV)[:,1]

    # CALC METRICS
    valid_roc = roc_auc_score(yvalid_CV, ypred)
    print(f"FOLD ROC: {valid_roc}")
    best_score_list.append(valid_roc)


print('#'*25)    
print(f'\n MEAN ROC: {np.mean(best_score_list)}')
print('#'*25)


#%% Blend with logistic regression
l2mod = LogisticRegression()
# Internal conversion of pandas is weird, change to numpy 
l2mod.fit(oof_train, df_y.to_numpy().ravel())
ypred = l2mod.predict_proba(pred_features)
# Check order of classes
#print(l2mod.classes_)
print(ypred)

# Make submission file
df_sub = df_test[['id']].copy()
df_sub['loan_status'] = ypred[:,1]

#%% Blend with lasso
l2mod = Lasso(alpha=0.01)
# Internal conversion of pandas is weird, change to numpy 
l2mod.fit(oof_train, df_y.to_numpy().ravel())
ypred = l2mod.predict(pred_features)
# Check order of classes
#print(l2mod.classes_)
display(ypred)

# Make submission file
df_sub = df_test[['id']].copy()
df_sub['loan_status'] = ypred[:]

# %%
# SAVE PREDICTION
VER = '23'
df_sub.to_csv(f"submissions/stack_{VER}.csv",index=False)

df_check = pd.read_csv(f'submissions/stack_{VER}.csv')
display(df_check)
print(df_check.shape)

# %%
