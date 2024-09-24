import torch
import numpy as np
import matplotlib.pylab as plt
import math
from sklearn.datasets import make_blobs

class LogisticRegression(torch.nn.Module):
    def __init__(self,n_features):
        super(LogisticRegression,self).__init__()
        #传入特征数量和一个输出结果
        self.linear = torch.nn.Linear(n_features,1)
    
    def forward(self,X):
        h = torch.sigmoid(self.linear(X))
        return h
    

if __name__ == "__main__":
    X,Y = make_blobs(n_samples=50, centers=2, random_state=42,cluster_std=0.5)
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

    #
    X_tensor = torch.tensor(X,dtype=torch.float32)
    Y_tensor = torch.tensor(Y,dtype=torch.float32)

    m = len(X)
    n = 2
    alpha = 0.001
    iterations = 20000
    #创建模型
    model = LogisticRegression(n)
    #创建二分类交叉熵损失函数
    criterion = torch.nn.BCELoss()
    #创建优化器
    optimizer = torch.optim.SGD(model.parameters(),lr=alpha)

    for epoch in range(iterations):
        y_pred = model(X_tensor).squeeze()
        loss = criterion(y_pred,Y_tensor)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 1000 == 0:
            print("epoch[%d]`s loss is %.3lf" % (epoch,loss.item() ))

    theta = list()
    for p in model.parameters():
        #flatten,无论是二维的还是一维的，都展开成一维的
        theta.extend(p.detach().numpy().flatten())
    
    w1 = theta[0]
    w2 = theta[1]
    b = theta[2]
    x1 = np.linspace(-4,10,100)
    x2 = - (w1 * x1 + b) * 1.0 / w2
    plt.plot(x1,x2)
    plt.show()   