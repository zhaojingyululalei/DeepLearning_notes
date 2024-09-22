import draw_math_graph as dmg
import torch
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2+x+5

X = torch.linspace(0,10,100,requires_grad=True)
print(X)
Y = f(X)
Y_sum = Y.sum().backward()

X_grad = X.grad.detach().numpy()
X = X.detach().numpy()
Y = Y.detach().numpy()
plt.plot(X,Y)
plt.title("orign VS grad")
plt.plot(X,X_grad)
plt.legend()
plt.show()

