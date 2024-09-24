import torch
import numpy as np
import matplotlib.pylab as plt
#n表示特征个数 x[0]=1 theta[0]=b
def hypothesis(theta,x,n):
    h = 0.0
    for i in range(n+1):
        h += theta[i] + x[i]
    return h 

def gradient_thetaj(X,Y,theta,n,m,j):
    sum = 0.0
    for i in range(m):
        h = hypothesis(theta,X[i],n)
        sum += (h-Y[i]) * X[i][j]
    return sum / m

def gradient_descent(X,Y,n,m,alpha,iterations):
    theta = [0]*(n+1)
    for i in range(iterations):
        for j in range(n+1):
            theta[j] = theta[j] - alpha * gradient_thetaj(X,Y,theta,n,m,j)

    return theta

def costJ(X,Y,theta,n,m):
    sum = 0.0
    for i in range(m):
        h = hypothesis(theta,X[i],n)
        sum += (h-Y[i]) *  (h-Y[i])

    return sum / (2*m)

if __name__ == '__main__':
    X = [[1.0,2.5, 3.0, 1.8, 5.6, 4.2, 2.1, 3.5, 4.8],
              [1.0,1.2, 2.7, 3.8, 4.5, 2.3, 1.5, 4.0, 3.2],
              [1.0,3.1, 4.2, 2.9, 5.1, 3.8, 2.7, 4.5, 5.0],
              [1.0,2.0, 3.5, 1.5, 4.8, 3.2, 2.0, 3.8, 4.2],
              [1.0,1.8, 2.5, 3.2, 4.0, 2.8, 1.9, 3.9, 3.5],
              [1.0,2.7, 3.8, 2.3, 5.0, 3.5, 2.5, 4.2, 4.6]]
    Y = [0, 1, 0, 1, 0, 1]

    m = len(X)
    n = 8
    theta = gradient_descent(X,Y,n,m,alpha=0.0001,iterations=3000)
    costj = costJ(X,Y,theta,n,m)
    print("costj = %.3lf"%(costj))


