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


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
                            roc_auc_score, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import train_test_split


from captum.attr import IntegratedGradients, LayerConductance, \
        NeuronConductance, LayerIntegratedGradients, configure_interpretable_embedding_layer, \
        remove_interpretable_embedding_layer
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import AUROC, BinaryAUROC

import gc
torch.cuda.empty_cache()
gc.collect()

import ctypes
# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

clean_memory()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()
os.chdir('/home/patrick/Python/timeseries/weather/kaggle/s4e10_Loan_Approval')


#%% Torch classes and model

class EmbDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.long)
         
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self,idx, batch_size):
        cat_val = self.cat_features[idx:idx+batch_size,:]
        num_val = self.num_features[idx:idx+batch_size,:]
        X_out   = torch.concat([num_val, cat_val], axis=1)
        y       = self.dfy[idx:idx+batch_size]
        return [num_val, cat_val, y]


class StdDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self, idx):
        cat = self.cat_features[idx]
        num = self.num_features[idx]
        y = self.dfy[idx]
        return [num, cat, y]


class FastDataLoader:
    def __init__(self, ds, batch_size=32):

        self.ds = ds
        self.dataset_len = ds.__len__()
        self.batch_size = batch_size

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = self.ds.__getitem__(self.i, self.batch_size)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)           


#%% Network
#%% Tabtransformer model

class Preprocessor(nn.Module):
    def __init__(self, numerical_columns: list, categorical_columns: list,
                 encoder_categories: dict, emb_dim: int, batchsize: int):
        super().__init__()
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.encoder_categories = encoder_categories
        self.emb_dim = emb_dim
        self.embed_layers = nn.ModuleDict()
        self.batchsize = batchsize
        
        for category in categorical_columns:
            embedding = nn.Embedding(num_embeddings=self.encoder_categories[category],
                                     embedding_dim=self.emb_dim)
            self.embed_layers[category] = embedding
    
    def forward(self, x_num_in, x_cat_in):
        x_nums = x_num_in
        # x_nums = []
        # for numerical in self.numerical_columns:
        #     x_num = torch.unsqueeze(x[numerical], dim=1)
        #     x_nums.append(x_num)
        # if len(x_nums) > 0:
        #     x_nums = torch.cat(x_nums, dim=1)
        # else:
        #     x_nums = torch.tensor(x_nums, dtype=torch.float32)
            
        x_cats = []
        for i, category in enumerate(self.categorical_columns):
            x_cat = self.embed_layers[category](x_cat_in[:,i])
            x_cats.append(x_cat)
        if len(x_cats) > 0:
            x_cats = torch.cat(x_cats, dim=1)
            x_cats = x_cats.reshape(len(x_cat_in), len(self.categorical_columns), self.emb_dim)
        else:
            x_cats = torch.tensor(x_cats, dtype=torch.float32)
        
        return x_nums, x_cats



class MLPBlock(nn.Module):
    def __init__(self, n_features, hidden_units, dropout_rates):
        super().__init__()
        self.mlp_layers = nn.Sequential()
        num_features = n_features
        for i, units in enumerate(hidden_units):
            self.mlp_layers.add_module(f'norm_{i}', nn.BatchNorm1d(num_features))
            self.mlp_layers.add_module(f'dense_{i}', nn.Linear(num_features, units))
            self.mlp_layers.add_module(f'act_{i}', nn.SELU())
            self.mlp_layers.add_module(f'dropout_{i}', nn.Dropout(dropout_rates[i]))
            num_features = units
    
    def forward(self, x):
        y = self.mlp_layers(x)
        return y
    

class TabTransformerBlock(nn.Module):
    def __init__(self, num_heads, emb_dim, attn_dropout_rate, ff_dropout_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, 
                        dropout=attn_dropout_rate, batch_first=True)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Dropout(ff_dropout_rate),
            nn.Linear(emb_dim*4, emb_dim))
        
    def forward(self, x_cat):
        attn_output , attn_weights = self.attn(x_cat, x_cat, x_cat)
        x_skip_1 = x_cat + attn_output
        x_skip_1 = self.norm_1(x_skip_1)
        feedforward_output = self.feedforward(x_skip_1)
        x_skip_2 = x_skip_1 + feedforward_output
        x_skip_2 = self.norm_2(x_skip_2)
        return x_skip_2
        

class TabTransformer(nn.Module):
    def __init__(self, numerical_columns, categorical_columns, num_transformer_blocks,
                 num_heads, emb_dim, attn_dropout_rates, ff_dropout_rates, 
                 mlp_dropout_rates, mlp_hidden_units_factors):
        super().__init__()
        self.transformers = nn.Sequential()
        for i in range(num_transformer_blocks):
            self.transformers.add_module(f'transformer_{i}', TabTransformerBlock(
                num_heads, emb_dim, attn_dropout_rates[i], ff_dropout_rates[i]))
        
        self.flatten = nn.Flatten()
        self.num_norm = nn.LayerNorm(len(numerical_columns))
        
        self.n_features = (len(categorical_columns)*emb_dim) + len(numerical_columns)
        mlp_hidden_units = [int(factor + self.n_features) for factor in mlp_hidden_units_factors]
        
        self.mlp = MLPBlock(self.n_features, mlp_hidden_units, mlp_dropout_rates)
        self.final_dense = nn.Linear(mlp_hidden_units[-1], 2)
        #self.final_sigmoid = nn.Sigmoid()
        
    def forward(self, x_nums, x_cats):
        contextualized_x_cats = self.transformers(x_cats)
        contextualized_x_cats = self.flatten(contextualized_x_cats)
        
        if x_nums.shape[-1] > 0:
            x_nums = self.num_norm(x_nums)
            features = torch.cat((x_nums, contextualized_x_cats), -1)
        else:
            features = contextualized_x_cats
        mlp_output = self.mlp(features)
        model_output = self.final_dense(mlp_output)
        #output = self.final_sigmoid(model_output)
        return model_output
    
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

#%%
train_ord = df_train.copy(deep=True)
test_ord = df_test.copy(deep=True)
FEATURE_CATS = CATS

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,)
train_ord[FEATURE_CATS] = oe.fit_transform(train_ord[FEATURE_CATS])
test_ord[FEATURE_CATS] = oe.transform(test_ord[FEATURE_CATS])

# Split y
y = train_ord.pop(TARGET[0])

#%%

def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data



#%%

def train(model, loader, optimizer, criterion, DEVICE):
    running_loss = 0.0
    model.train()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        output = model.forward(in1, in2)
        
        loss = criterion(torch.flatten(output), label.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    training_loss = running_loss/len(loader)
    return training_loss


def valid(model, loader, DEVICE, oof, val_idx):
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        
        output = model.forward(in1, in2)
        loss = criterion(torch.flatten(output), label.float())
        running_loss += loss.item()
        
        y_prediction.append(torch.flatten(output).detach().cpu().tolist())
        y_true.append(label.detach().cpu().tolist())
    
    # Flassten prediction and labels    
    y_true1 = np.array([v for lst in y_true for v in lst])
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])
    # Get oof for this fold
    oof[val_idx] = y_prediction1
    
    validation_loss = running_loss/len(loader)
    validation_aucroc = roc_auc_score( y_true1, y_prediction1 )
    return validation_loss, validation_aucroc, oof

def test_predictions(model, loader, DEVICE):
    y_prediction = []
    model.eval()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        output = model.forward(in1, in2)
        
        y_prediction.append(torch.flatten(output).detach().cpu().tolist())
        
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])
    return y_prediction1


#%%
def plot_data(train_d, valid_d):
    xx = np.arange(len(train_d))
    plt.figure()
    plt.plot(xx, train_d, label='Train')
    plt.plot(xx, valid_d, label='Validation')
    plt.legend()
    plt.show()
    return

df_X = train_ord.copy()
df_y = y.copy()
test_X = test_ord.copy()
meta_data = {}
meta_data['ORD'] = ORD
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(ORD)
# Use category for embedding
meta_data = get_postsplit_meta(df_X, meta_data)

FOLDS = 5
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
PATIENCE = 3
DEVICE = torch.device('cuda:0')

ydummy = pd.DataFrame(data=np.zeros((df_test.shape[0],1)), columns=['loan_status']) 
testdataset = EmbDataset(test_X, ydummy, meta_data['ORD'], meta_data['CATS'])
testloader = FastDataLoader(testdataset, batch_size=BATCH_SIZE)

#kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(df_X))
fold_train_score = []
for i, (train_idx, val_idx) in enumerate(kf.split(train_ord, df_y)):
    print('#'*25)
    print(f' FOLD {i} \n')
    # SET UP DATA SPLITS
    Xtrain_CV = df_X.iloc[train_idx].copy()
    ytrain_CV = df_y.iloc[train_idx].copy()
    
    Xvalid_CV = df_X.iloc[val_idx].copy()
    yvalid_CV = df_y.iloc[val_idx].copy()

    test_CV = test_X.copy()

    # SET UP DATA
    traindataset = EmbDataset(Xtrain_CV, ytrain_CV, meta_data['ORD'], meta_data['CATS'])
    validdataset = EmbDataset(Xvalid_CV, yvalid_CV, meta_data['ORD'], meta_data['CATS'])
    trainloader = FastDataLoader(traindataset, batch_size=BATCH_SIZE)
    validloader = FastDataLoader(validdataset, batch_size=BATCH_SIZE)

    # SET UP DATA
    # traindataset = StdDataset(Xtrain_CV, ytrain_CV, meta_data['ORD'], meta_data['CATS'])
    # validdataset = StdDataset(Xvalid_CV, yvalid_CV, meta_data['ORD'], meta_data['CATS'])
    # trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE)
    # validloader = DataLoader(validdataset, batch_size=BATCH_SIZE)

    
    # DEF MODEL
    model = Model(meta_data, 0.1, 8*[64], 8*[0.1]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=PATIENCE)
    #lr_scheduler = LRScheduler(optimizer)

    train_epoch_list = []
    valid_epoch_list = []
    start_time = time.time()
    for epoch in range(EPOCHS):
        train_data = train(model, trainloader, optimizer, criterion, DEVICE)
        validation_loss, validation_aucroc, oof = valid(model, validloader, DEVICE, oof, val_idx)
        print(f'Epoch: {epoch}/{EPOCHS}, Train loss: {train_data:.6f}, Validation loss: {validation_loss:.6f}, Validation ROC_AUC: {validation_aucroc:.6f}')
        train_epoch_list.append(train_data)
        valid_epoch_list.append(validation_loss)
    
        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Eary stopping")
            break
    early_stopping.load_best_model(model)
    
    test_pred = test_predictions(model, testloader, DEVICE)
    if i == 0:
        pred = test_pred
    else:
        pred += test_pred
            
    fold_train_score.append(validation_aucroc)
    plot_data(train_epoch_list, valid_epoch_list)



fold_mean = np.mean(fold_train_score)
print(f'Mean ROC: {fold_mean}')
pred /= FOLDS
end_time = time.time()
print(f'Total time: {end_time - start_time}')
   
# %%

print(oof.shape)
print(pred.shape)

print(pred)

# %%
