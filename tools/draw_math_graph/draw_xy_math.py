import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import torch

    
def draw_3D_expression(expression_str,title,xlabel,ylabel,zlabel):
    # 解析表达式
    x, y = sp.symbols('x y')
    expression = sp.sympify(expression_str)
    
    # 生成数据
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # 使用sympy的lambdify将表达式转换为函数
    f = sp.lambdify((x, y), expression, 'numpy')
    
    # 计算Z值
    Z = f(X, Y)
    
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制表面
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    # 添加颜色条
    fig.colorbar(surf)
    
    # 设置标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    # 显示图形
    plt.show()

def draw_decision_boundray(minx1,maxx1,minx2,maxx2,model):
    xx1,xx2 = np.meshgrid(np.arange(minx1,maxx1,0.02),
                          np.arange(minx2,maxx2,0.02))
    
    x1s = xx1.ravel()
    x2s = xx2.ravel()

    z = list()
    for x1,x2 in zip(x1s,x2s):
        test_point=torch.FloatTensor([[x1,x2]])
        output = model(test_point) #返回的是未经过softmax处理的输出
        _,predicted = torch.max(output,1)#
        z.append(predicted.item())

    z = np.array(z).reshape(xx1.shape)
    return xx1,xx2,z
if __name__ == "__main__":
    x = list()
