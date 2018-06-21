# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

import attention_example
import lstm_example

from tqdm import trange


max_iter = 100
xs = range(max_iter)
lstm_loss = []
attn_loss = []
for i in trange(max_iter):
    lstm_loss.append(lstm_example.step())
    attn_loss.append(attention_example.step())
plt.plot(xs, lstm_loss, 'b-', label='lstm')
plt.plot(xs, attn_loss, 'r-', label='attn')
plt.legend()
plt.show()
