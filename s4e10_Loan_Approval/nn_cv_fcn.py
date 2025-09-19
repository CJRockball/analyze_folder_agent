
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import AUROC, BinaryAUROC



# Torch classes and model

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


#  Create NN model
# Put all into embedding layer, add dropout
# Concatenate, with num feature (314)
# BN
# Dense layer (314->256)
# Attention, concatenate 314 and 256
# Dense layer (570->128)
# Dropout (0.3)
# Dene(128->1)
# Sigmoid


class Model(nn.Module):
    def __init__(self, meta_data, emb_dropout, d_out=1):
        super().__init__()
        num_size = meta_data['num_nums']
        emb_sizes = meta_data['emb_sizes']
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_sizes])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
            #nn.init.kaiming_normal_(emb.weight.data)
          
        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        # Calculate in_features to linear layer
        emb_vector_sum = sum([e.embedding_dim for e in self.embedding_d])
        # Input for dense layerm; no out from emb, no num features
        emb_out = emb_vector_sum
        
        # Initialize fc layers
        self.fc0 = nn.Linear(num_size, num_size)
        self.fc1 = nn.Linear(emb_out+num_size, 256)
        self.fc2 = nn.Linear(562, 128)
        self.fc3 = nn.Linear(128, d_out)
        # Initialize Batch Norm 
        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(emb_out+num_size)
        # Dropout
        self.dp1 = nn.Dropout(0.3)
    
    
    def forward(self, num_fields, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        x1 = x1 + [num_fields]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        x1 = self.emb_dropout(x1)
        
        # Dropout for embeddings
        x1 = self.bn1(x1)
        x2 = F.relu(self.fc1(x1))
        x3 = torch.cat([x1, x2], 1)
        
        x3 = F.relu(self.fc2(x3))
        x3 = self.dp1(x3)
        x3 = self.fc3(x3)
        out = F.sigmoid(x3) # Sigmoid because we're using BCELoss
        return out

# Embedding sizes
def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''    
    test_emb = [( len(Xtrain[col].unique()), int(min(128,1.6 * len(Xtrain[col].unique())**0.56)) ) for col in meta_data['CATS'] ]
    meta_data['emb_sizes'] = test_emb
    
    return meta_data

# Training, Validation, Prediction
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


def valid(model, loader, criterion, DEVICE, oof, val_idx):
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


# Main training loop
def plot_data(train_d, valid_d):
    xx = np.arange(len(train_d))
    plt.figure()
    plt.plot(xx, train_d, label='Train')
    plt.plot(xx, valid_d, label='Validation')
    plt.legend()
    plt.show()
    return

def nn_algo(df_train, y, df_test, NUMS, CATS):
    df_X = df_train.copy()
    df_y = y.copy()
    test_X = df_test.copy()
    meta_data = {}
    meta_data['ORD'] = NUMS
    meta_data['CATS'] = CATS
    meta_data['num_cats'] = len(CATS)
    meta_data['num_nums'] = len(NUMS)
    # Use category for embedding
    df_all = pd.concat([df_X, test_X])
    meta_data = get_postsplit_meta(df_all, meta_data)

    FOLDS = 5
    EPOCHS = 20
    LR = 3e-4
    BATCH_SIZE = 128
    PATIENCE = 3
    DEVICE = torch.device('cpu') #'cuda:0')

    ydummy = pd.DataFrame(data=np.zeros((df_test.shape[0],1)), columns=['loan_status']) 
    testdataset = EmbDataset(test_X, ydummy, meta_data['ORD'], meta_data['CATS'])
    testloader = FastDataLoader(testdataset, batch_size=BATCH_SIZE)

    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    oof = np.zeros(len(df_X))
    fold_train_score = []
    for i, (train_idx, val_idx) in enumerate(kf.split(df_X, df_y)):
        print('#'*25)
        print(f' FOLD {i} \n')
        # SET UP DATA SPLITS
        Xtrain_CV = df_X.iloc[train_idx].copy()
        ytrain_CV = df_y.iloc[train_idx].copy()
        
        Xvalid_CV = df_X.iloc[val_idx].copy()
        yvalid_CV = df_y.iloc[val_idx].copy()

        test_CV = test_X.copy()

        # SET UP DATA special dataset, dataloader functions
        traindataset = EmbDataset(Xtrain_CV, ytrain_CV, meta_data['ORD'], meta_data['CATS'])
        validdataset = EmbDataset(Xvalid_CV, yvalid_CV, meta_data['ORD'], meta_data['CATS'])
        trainloader = FastDataLoader(traindataset, batch_size=BATCH_SIZE)
        validloader = FastDataLoader(validdataset, batch_size=BATCH_SIZE)

        # SET UP DATA standard dataset, dataloader functions
        # traindataset = StdDataset(Xtrain_CV, ytrain_CV, meta_data['ORD'], meta_data['CATS'])
        # validdataset = StdDataset(Xvalid_CV, yvalid_CV, meta_data['ORD'], meta_data['CATS'])
        # trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE)
        # validloader = DataLoader(validdataset, batch_size=BATCH_SIZE)

        
        # DEF MODEL
        model = Model(meta_data, 0.3).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR) #, weight_decay=0.0001)
        early_stopping = EarlyStopping(patience=PATIENCE)
        #lr_scheduler = LRScheduler(optimizer)

        train_epoch_list = []
        valid_epoch_list = []
        start_time = time.time()
        for epoch in range(EPOCHS):
            train_data = train(model, trainloader, optimizer, criterion, DEVICE)
            validation_loss, validation_aucroc, oof = valid(model, validloader, criterion, DEVICE, oof, val_idx)
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
    return


