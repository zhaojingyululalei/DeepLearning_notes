import torch
import numpy as np
import matplotlib.pyplot as plt

def linear_model(x,w,b):
    return x * w + b

#h预测值，y真实值
def mse_loss(h,y):
    return torch.mean((h-y) ** 2)

if __name__ == '__main__':
    X = torch.tensor([50.0, 60.0 , 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    Y = torch.tensor([280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0])
    #形状为 (1,)  并计算梯度
    w = torch.randn(1,requires_grad=True)   
    b = torch.randn(1,requires_grad=True)

    alpha = 0.0001
    iterations= 1000

    for i in range(iterations):
        h = linear_model(X,w,b)
        loss = mse_loss(h,Y)
        loss.backward()

        w.data -= alpha * w.grad.data
        b.data -= alpha * b.grad.data

        #w b 移动到了新的位置，计算新的梯度。梯度清0
        w.grad.zero_()
        b.grad.zero_()

        if i % 100 == 0:
            print("epoch[%d],loss :%.3lf" %(i,loss.data))