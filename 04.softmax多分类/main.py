import torch
import torch.nn as nn
from make_data import make_data 
import draw_math_graph as dmg
import matplotlib.pyplot as plt
import numpy as np

class SoftmaxRegression(nn.Module):
    def __init__(self, n_features, n_classes) -> None:
        #调用父类构造函数
        super(SoftmaxRegression,self).__init__()
        #返回一个全连接层对象
        self.linear = nn.Linear(n_features,n_classes)
    def forward(self,x):
        #在 PyTorch 中，线性层对象实现了 __call__ 方法，使得它可以像函数一样被调用
        #返回的是一个张量，目前的size是1*n_classes
        return self.linear(x)
    



#绘制样本
k = 3
k_num = [100,100,100]
k_color = ["green","red","blue"]
k_center = [
        [0,-2],
        [-2,2],
        [2,2]
    ]
samples = make_data(k,k_num,k_center)
dmg.draw_coordinate_system(4,-4,-4,4,"test","x","y")

for i in range(k):
    plt.scatter(samples[i][:,0],samples[i][:,1])
    


n_features = 2
n_classes = 3
n_epochs = 10000
learning_rate = 0.01

green = torch.FloatTensor(samples[0])
blue = torch.FloatTensor(samples[1])
red = torch.FloatTensor(samples[2])
data = torch.cat((green,blue,red),dim=0)

label = [0]*len(green) + [1]*len(blue) + [2] * len(red) #数组拼接，numpy才支持运算
#test = np.array([1,2,3,4]) + np.array([1,2,3,4])
label = torch.LongTensor(label)

model = SoftmaxRegression(n_features,n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_epochs):
    output = model(data) #forward，返回每个样本属于每个类别的预测概率
    #将每个样本的输出概率(1*n_classes)和标签都传入，计算损失
    loss = criterion(output,label)
    loss.backward()  #计算梯度
    optimizer.step() #更新参数w b
    optimizer.zero_grad()#梯度清0

    if epoch %1000 ==0:
        print("%d iteration: loss is %.3lf" % (epoch,loss.item()))

xx1,xx2,z = dmg.draw_decision_boundray(-4,4,-4,4,model)

plt.contour(xx1,xx2,z,colors = ["orange"])
plt.show()

