'''
run stgcn
'''

from stgcn import *
from load_data_stgcn import *


import os
import zipfile
import numpy as np
import torch
import pickle
import scipy.sparse as sp

matrix_path = "/home/cseadmin/yindu/github/data/stgcndata/W_pemsbay.csv"
data_path = "/home/cseadmin/yindu/github/data/stgcndata/V_pemsbay.csv"
save_path = "/home/cseadmin/yindu/github/save/stgcnmodel-p.pt"


'''
parameters
'''
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

day_slot = 288
n_train, n_val, n_test = 109, 36, 36

n_his = 12
n_pred = 3
n_route = 325
Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0

batch_size = 128
epochs = 100
lr = 1e-3

torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


#graph
W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)

#Standardization
train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

#Transform Data
x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

#DataLoader
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

#Loss & Model & Optimizer
loss = nn.MSELoss()
model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

#LR Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

#Training & Save Model¶
min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

#Load Best Model
best_model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
best_model.load_state_dict(torch.load(save_path))

#Evaluation
l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)