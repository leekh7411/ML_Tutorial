import torch
import numpy as np
w = 1.0
def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

for _w in np.arange(0.0,4.0,0.1):
    print("W = " , _w)
    w = _w
    l_sum = 0
    for x_val , y_val in zip(x_data,y_data):
        y_pred = forward(x_val)
        l = loss(x_val,y_val)
        l_sum += l
        print('\t',x_val,y_val,y_pred,l)
    print("MSE: ", l_sum / 3)