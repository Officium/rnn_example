# -*- coding: utf-8 -*-
from itertools import chain
import warnings

import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")


sample_num = 8000
sample_len = 10
hidden_size = 64 - 8
train_x = np.random.randint(0, 64 - 8, (sample_num, sample_len))
x_dim = 64 - 8
###############################################################################################
encoder = nn.LSTMCell(input_size=x_dim, hidden_size=hidden_size)
decoder = nn.LSTMCell(input_size=x_dim, hidden_size=hidden_size)
output = nn.Sequential(
    nn.Linear(in_features=hidden_size, out_features=x_dim),
    nn.LogSoftmax()
)
nn.init.xavier_normal_(output[0].weight)
nn.init.constant_(output[0].bias, 0)
optim = torch.optim.Adam(chain(
    encoder.parameters(), decoder.parameters(), output.parameters()
))
###############################################################################################


def step():
    hx, cx = torch.randn(len(train_x), hidden_size), torch.randn(len(train_x), hidden_size)
    for i in range(sample_len):
        x = torch.zeros(sample_num, x_dim).scatter_(1, torch.Tensor(train_x[:, [i]]).long(), 1)
        hx, cx = encoder(x.float(), (hx, cx))
    loss = 0
    for i in range(sample_len):
        if i == 0:
            x = torch.zeros(sample_num, x_dim)  # EOS
        else:
            x = torch.zeros(sample_num, x_dim).scatter_(1, torch.Tensor(train_x[:, [i-1]]).long(), 1)
        hx, cx = decoder(x.float(), (hx, cx))
        loss += -output(hx).gather(1, torch.Tensor(train_x[:, [i]]).long()).mean() / sample_len

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


if __name__ == '__main__':
    for i_iter in range(1000):
        iter_loss = step()
        print('Iter {}, Loss {:.4f}'.format(i_iter, iter_loss))
