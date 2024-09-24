import matplotlib.pyplot as plt
import numpy as np

#m样本个数  x单个样本  X所有样本  y单个标签，Y所有标签
def gradient_theta0(X,Y,theta0,theta1):
    sum = 0.0
    m = len(X)
    for i in range(m):
        sum += theta0 + theta1 * X[i] - Y[i]
    return sum / m

def gradient_theta1(X,Y,theta0,theta1):
    sum = 0.0
    m = len(X)
    for i in range(m):
        sum += (theta0 + theta1 * X[i] - Y[i]) + X[i]
    return sum / m

def gradient_descent(X,Y,alpha,iterations):
    theta0 = 0.0
    theta1 = 0.0
    for i in range(iterations):
        theta0 = theta0 - alpha * gradient_theta0(X,Y,theta0,theta1)
        theta1 = theta1 - alpha * gradient_theta1(X,Y,theta0,theta1)

    return theta0,theta1

def costJ(X,Y,theta0,theta1):
    sum = 0.0
    m = len(X)
    for i in range(m):
        sum += (theta0 + theta1 * X[i] - Y[i]) * (theta0 + theta1 * X[i] - Y[i])
    return sum / (2 * m)

def predict(theta0,theta1,x):
    return theta0 + theta1 * x

if __name__ == '__main__':
    X = [50.0, 60.0 , 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    Y = [280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0]
    alpha = 0.0001
    iterations = 30000
    theta0,theta1 = gradient_descent(X,Y,alpha,iterations)
    cost = costJ(X,Y,theta0,theta1)
    print("after 3000 iterations,cost = %.3lf" %(cost))
    print("predict(112)=%.3lf" %(predict(theta0,theta1,112)))
    print("predict(158)=%.3lf" %(predict(theta0,theta1,158)))