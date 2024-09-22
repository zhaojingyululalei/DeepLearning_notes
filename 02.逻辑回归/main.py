import torch
import numpy as np
import matplotlib.pylab as plt
import math
from sklearn.datasets import make_blobs

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


if __name__ == "__main__":
    X,y = make_blobs(n_samples=50, centers=2, random_state=42,cluster_std=0.5)
    pos_x1 = list()
    pos_x2 = list()
    neg_x1 = list()
    neg_x2 = list()
    
    for i in range(len(y)):
        if  y[i] == 0 :
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
   # plt.show()

    m = len(X)
    n = 2
    alpha = 0.001
    iterate = 20000

    X = np.insert(X,0,values=[1]*m,axis=1).tolist()
    y = y.tolist()

    theta = gradient_descent(X,y,n,m,alpha,iterate)
    costJ = costj(X,y,theta,n,m) 
    for i in range(len(theta)):
        print("theta[%d]=%lf" % (i,theta[i]))
    print("cost=%lf" % (costJ))

#绘制决策边界
    w1 = theta[1]
    w2 = theta[2]
    b = theta[0]

    x1 = np.linspace(-4,10,100)
    x2 = -1.0 * (w1*x1+b) / w2
    plt.plot(x1,x2)
    plt.show()