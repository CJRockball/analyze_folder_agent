#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df_train = pd.read_csv('data/raw/train.csv')
df_train = df_train.drop(columns=['id'])
df_org = pd.read_csv('data/raw/credit_risk_dataset.csv')
df_test = pd.read_csv('data/raw/test.csv')
df_test = df_test.drop(columns=['id'])

# %%

display(df_train.head(4))
# print(df_train.isnull().sum())
# print(df_train.shape)
# print(df_train.columns)
print(df_train.info())

# %%

CATS = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
ORD = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income', 'cb_person_cred_hist_length']
TARGET = ['loan_status']

print(len(CATS+ORD))
print(len(df_train.columns))

# %% Plot train_data distribution
# ------------------- CATS

for col in CATS+TARGET:
    df_train[col].value_counts().plot(kind='barh')
    plt.show()

# %% ------------- NUMS
# Check age

for col in ORD:
    df_train[col].plot(kind='hist')
    plt.title(col)
    plt.show()
    
#df_train.person_age.max()
#df_train.loc[df_train['person_age'] > 60., ['person_age']].plot(kind='hist')


# %% NUMBER OF FEATURES

for name in CATS:
    print(name, len(df_train[name].unique()))

# %% Compare distributions between datasets train, org, test
# Compare distributions by checking CDF and KS
import seaborn as sns
from scipy.stats import ks_2samp

for col_name in ORD:
    print('#'*25)
    print(col_name, '\n')
    a = np.concatenate((df_train.loc[:,col_name], df_org.loc[:,col_name], df_test.loc[:,col_name]))
    df = pd.DataFrame({col_name:a, 'set':['training']*len(df_train)+['org']*len(df_org) +['test']*len(df_test)})
    sns.ecdfplot(data=df, x=col_name, hue='set')
    plt.show()

for col_name in ORD:
    print(f"{col_name}:, {ks_2samp(df_train.loc[:,col_name], df_test.loc[:,col_name]).statistic:.6f}")
    
#%% Check correlations
# Pearson, SPearman, Mutual Information, PhiK
no_features = len(df_train.loc[:,ORD].columns)
p_corr = df_train.loc[:,ORD].corr('kendall')

fig, ax = plt.subplots()
im = ax.imshow(p_corr.values)
fig.colorbar(im)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(no_features), labels=df_train.loc[:,ORD].columns)
ax.set_yticks(np.arange(no_features), labels=df_train.loc[:,ORD].columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(no_features):
    for j in range(no_features):
        text = ax.text(j, i, round(p_corr.to_numpy()[i, j],2),
                       ha="center", va="center", color="w")
        
        

# %%


