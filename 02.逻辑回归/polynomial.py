import torch
import numpy as np
import matplotlib.pylab as plt
import math
from sklearn.datasets import make_gaussian_quantiles #生成非线性数据

#theta --参数   x --某个样本  n --样本的特征个数 m --样本数量
# z  h  预测值
# j 第几个特征
#alpha 迭代步长
# iterate 迭代次数
def sigmod(z):
    return 1.0 / (1 + math.exp(-1 * z ))


def hypothesis(theta,x,n):
    z = 0.0
    for i in range(n+1):  #i ~ [0,n]
        z += theta[i] * x[i]
    return sigmod(z)

def gradient_thetaj(X,y,theta,n,m,j):
    sum = 0.0
    for i in range(m):
        h = hypothesis(theta,X[i],n)
        sum += (h-y[i]) * X[i][j]
    return sum / m

def gradient_descent(X,y,n,m,alpha,iterate):
    #初始化theta  n个特征和一个偏置项
    theta = [0] * (n + 1)  
    for i in range(iterate):
        for j in range(n+1):
            theta[j] = theta[j] - alpha * gradient_thetaj(X,y,theta,n,m,j)
    return theta

def costj(x,y,theta,n,m):
    sum = 0.0
    for i in range(m):
        h = hypothesis(theta,x[i],n)
        sum += -1*y[i] * math.log(h) - (1-y[i]) * math.log(1-h)
    return sum / m

def polynomial(X):
    polyX = list()
    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]
        polyX.append([1,x1,x2,x1*x1,x2*x2,x1*x2])
    
    return polyX

def make_counter_point(minx1,maxx1,minx2,maxx2,theta,n):
    xx1,xx2 = np.meshgrid(np.arange(minx1,maxx1,0.02),
                          np.arange(minx2,maxx2,0.02))
    x1s = xx1.ravel()
    x2s = xx2.ravel()
    X = list()
    m = len(x1s)
    for i in range(m):
        X.append([x1s[i],x2s[i]])
    polyx = polynomial(X)

    z = list()
    for i in range(m):
        h = hypothesis(theta,polyx[i],n)
        if h>0.5:
            z.append(1)
        else:
            z.append(0)
    z = np.array(z).reshape(xx1.shape)
    return xx1,xx2,z




if __name__ == "__main__":
    #画数据点
    X,Y = make_gaussian_quantiles(n_samples=30,n_features=2,n_classes=2,random_state=0)

    pos_x1 = list()
    pos_x2 = list()
    neg_x1 = list()
    neg_x2 = list()
    for i in range(len(Y)):
        if  Y[i] == 0 :
            neg_x1.append(X[i][0])
            neg_x2.append(X[i][1])
        else:
            pos_x1.append(X[i][0])
            pos_x2.append(X[i][1])    
    
    board = plt.figure()
    axis = board.add_subplot(1,1,1)
    axis.set(xlim=[-4,10],ylim=[-4,10],title="logstic regression",xlabel="x1",ylabel="x2")
    plt.scatter(pos_x1,pos_x2,color="blue",marker="o")
    plt.scatter(neg_x1,neg_x2,color="red",marker="x")
    
    #画决策边界    
    polyX = polynomial(X)
    m = len(polyX)
    n = 5
    alpha = 0.001
    iterations = 10000
    theta = gradient_descent(polyX,Y,n,m,alpha,iterations)
    costJ = costj(polyX,Y,theta,n,m)

    print("cost=%.3lf"%(costJ))

    xx1,xx2,z = make_counter_point(-4,10,-4,10,theta,n)
    plt.contour(xx1,xx2,z)
    plt.show()

