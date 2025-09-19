#%%

import sklearn
import pandas as pd
import numpy as np
import time 

from sklearn.model_selection import StratifiedKFold, cross_val_score, \
                RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import  RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.compose import make_column_transformer

import faiss

#%%

X = pd.read_csv('data/raw/train.csv', index_col='id')
y = X.pop('loan_status')

#%%

# FAISS kNN from this great post: https://www.kaggle.com/competitions/playground-series-s4e9/discussion/532329
class FaissKNeighborsRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_neighbors=100, weights='uniform', device='cpu'):
        self.n_neighbors = n_neighbors
        self.device = device
        self.weights = weights

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.index = faiss.IndexFlatL2(X.shape[1])
        if self.device == 'gpu':
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(X.astype(np.float32))
        self.y = y
        return self

    def predict(self, X):
        X = check_array(X).astype(np.float32)
        dist, I = self.index.search(X, self.n_neighbors)
        with np.errstate(divide='ignore'):
            if self.weights == 'uniform':
                dist = np.ones(dist.shape)
            elif self.weights == 'distance':
                dist = 1/dist
            else:
                assert hasattr(self.weights, '__call__')
                dist = self.weights(dist)
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return np.average(self.y[I], axis=1, weights=dist)


#%% Check kNN

start_time = time.time()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = make_scorer(roc_auc_score, response_method='predict')
p = 1.4
model = make_pipeline(
    TargetEncoder(random_state=0),
    FaissKNeighborsRegressor(n_neighbors=250, weights=lambda x:x**-p)
) 

scores = cross_val_score(model, X, y, scoring=scoring, cv=kfold, n_jobs=4)
print(F'CV AUC score: {np.mean(scores):.5f} ± {np.std(scores):.5f}')
end_time = time.time()
print(f'Time: {end_time - start_time}')

# %% Check how dropping a column affects score

start_time = time.time()
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring = make_scorer(roc_auc_score, response_method='predict')

base_model = RandomForestRegressor(random_state=42, n_jobs=-1, max_features='sqrt')
models = {
    'rfr': make_pipeline(
        TargetEncoder(random_state=42),
        base_model
    ),
    'drop col + rfr': make_pipeline(
        make_column_transformer(
            ('drop', ['cb_person_cred_hist_length']),
            remainder=TargetEncoder(random_state=42)
        ),
        base_model
    )
}

for m in models:
    scores = cross_val_score(
        models[m], X, y, scoring=scoring, cv=kfold, n_jobs=4
    )
    print(F'{m}: {np.mean(scores):.5f} ± {np.std(scores):.5f}')

end_time = time.time()
print(f'Time: {end_time - start_time}')

# %% RANDOM FORREST PIPELINE WITH CV 0.96
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder, FunctionTransformer
from cuml.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import warnings

X = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv', index_col='id')
y = X.pop('loan_status')

kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
scoring='roc_auc'

params = {
    'max_features': 2,
    'n_estimators': 2000,
    'min_samples_leaf': 3,
    'split_criterion': 'entropy',
    'n_bins': 1024,
    'random_state': 0
}

model = make_pipeline(
    TargetEncoder(random_state=0).set_output(transform='pandas'),
    FunctionTransformer(lambda X: X.astype(np.float32)),
    RandomForestClassifier(**params, n_streams=1, output_type='numpy')
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    %time scores = cross_val_score(model, X, y, scoring=scoring, cv=kfold, n_jobs=4)

print(F'CV AUC score: {np.mean(scores):.5f} ± {np.std(scores):.5f}')
