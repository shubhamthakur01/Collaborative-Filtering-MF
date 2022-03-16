import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["user", "item"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, seed=23):
        super(MF, self).__init__()
        torch.manual_seed(seed)
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # init 
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)

    def forward(self, u, v):
        ### BEGIN SOLUTION
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return torch.sigmoid((U*V).sum(1) +  b_u  + b_v)


def train_one_epoch(model, train_df, optimizer):
    """ Trains the model for one epoch"""
    model.train()
    users = torch.LongTensor(train_df.user.values)  #.cuda()
    items = torch.LongTensor(train_df.item.values) #.cuda()
    ratings = torch.FloatTensor(train_df.rating.values)  #.cuda()

    y_hat = model(users, items)
    loss = F.binary_cross_entropy(y_hat, ratings)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    return train_loss

def valid_metrics(model, valid_df):
    """Computes validation loss and accuracy"""
    model.eval()
    ### BEGIN SOLUTION
    users = torch.LongTensor(valid_df.user.values) # .cuda()
    items = torch.LongTensor(valid_df.item.values) #.cuda()
    ratings = torch.FloatTensor(valid_df.rating.values) #.cuda()
    y_hat = model(users, items)
    valid_loss = F.binary_cross_entropy(y_hat, ratings).item()
    y_hat_numpy = y_hat.detach().numpy()
    y_hat_numpy = np.where(y_hat_numpy > 0.5, 1, 0)
    #valid_acc = accuracy_score(ratings.numpy(), y_hat_numpy)
    return valid_loss, np.mean(ratings.numpy() ==y_hat_numpy)


def training(model, train_df, valid_df, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        train_loss = train_one_epoch(model, train_df, optimizer)
        valid_loss, valid_acc = valid_metrics(model, valid_df) 
        print("train loss %.3f valid loss %.3f valid acc %.3f" % (train_loss, valid_loss, valid_acc)) 
        
