#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression, Lasso ,Ridge,RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


#ada_oof = pd.read_csv('ensembling/ada_oof_basic.csv').rename(columns={'pred_basic':'pred_ada'})
cat_oof = pd.read_csv('ensembling/cat_oof_basic.csv').rename(columns={'pred_basic':'pred_cat'})
extra_tree_oof = pd.read_csv('ensembling/ExtraTree_oof_basic.csv').rename(columns={'pred_basic':'pred_et'})
gradbc_oof = pd.read_csv('ensembling/GradientBoostedClassifier_oof_basic.csv').rename(columns={'pred_basic':'pred_gbc'})
histgb_oof = pd.read_csv('ensembling/HistGradBoost_oof_basic.csv').rename(columns={'pred_basic':'pred_hgb'})
lgbm_oof = pd.read_csv('ensembling/lgbm_oof_basic.csv').rename(columns={'pred_basic':'pred_lgbm'})
randfor_oof = pd.read_csv('ensembling/RandomForest_oof_basic.csv').rename(columns={'pred_basic':'pred_rf'})
xgb_oof = pd.read_csv('ensembling/xgb_oof_basic.csv').rename(columns={'pred_basic':'pred_xgb'})

display(randfor_oof)

# Load train
df_X = pd.read_csv('data/raw/train.csv')
df_X = df_X.drop(columns=['id'])
df_X_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_X = pd.concat([df_X, df_X_org], axis=0)
df_y = df_X[['loan_status']]
# Load test
df_test = pd.read_csv('data/raw/test.csv')


# %%

oof_files = [extra_tree_oof,gradbc_oof,histgb_oof,
             randfor_oof,xgb_oof,cat_oof,lgbm_oof]
oof = pd.concat(oof_files, axis=1)
display(oof)


# %%

oof_corr = oof.corr()
features = oof_corr.columns
len_feat = len(features)

fig, ax = plt.subplots()
im = ax.imshow(oof_corr.values)
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
        text = ax.text(j, i, round(oof_corr.to_numpy()[i, j],2),
                       ha="center", va="center", color="w")
        


#%% Blend in CV


df_X_oof = oof.copy()
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

#%% Load prediction features
#ada_pred = pd.read_csv('ensembling/ada_pred_basic.csv').rename(columns={'loan_status':'pred_ada'})
cat_pred = pd.read_csv('ensembling/cat_pred_basic.csv').rename(columns={'loan_status':'pred_cat'})
extra_tree_pred = pd.read_csv('ensembling/ExtraTree_pred_basic.csv').rename(columns={'loan_status':'pred_et'})
gradbc_pred = pd.read_csv('ensembling/GradientBoostedClassifier_pred_basic.csv').rename(columns={'loan_status':'pred_gbc'})
histgb_pred = pd.read_csv('ensembling/HistGradBoost_pred_basic.csv').rename(columns={'loan_status':'pred_hgb'})
lgbm_pred = pd.read_csv('ensembling/lgbm_pred_basic.csv').rename(columns={'loan_status':'pred_lgbm'})
randfor_pred = pd.read_csv('ensembling/RandomForest_pred_basic.csv').rename(columns={'loan_status':'pred_rf'})
xgb_pred = pd.read_csv('ensembling/xgb_pred_basic.csv').rename(columns={'loan_status':'pred_xgb'})

pred_files = [extra_tree_pred,gradbc_pred,histgb_pred,
             randfor_pred,xgb_pred,cat_pred,lgbm_pred]
pred_features = pd.concat(pred_files, axis=1)
pred_features = pred_features.drop(columns=['id'])

#%% Blend with logistic regression
l2mod = LogisticRegression()
# Internal conversion of pandas is weird, change to numpy 
l2mod.fit(oof, df_y.to_numpy().ravel())
ypred = l2mod.predict_proba(pred_features)
# Check order of classes
#print(l2mod.classes_)
print(ypred)

# Make submission file
df_sub = df_test[['id']].copy()
df_sub['loan_status'] = ypred[:,1]
# %%

# SAVE PREDICTION
VER = '24'
df_sub.to_csv(f"submissions/stack_{VER}.csv",index=False)

df_check = pd.read_csv(f'submissions/stack_{VER}.csv')
display(df_check)
print(df_check.shape)

# %%
