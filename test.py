import numpy as np
import matplotlib.pyplot as plt

# 定义一维数组作为输入
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7])

# 使用meshgrid函数生成二维坐标点网格
X, Y = np.meshgrid(x, y)

# 打印生成的二维坐标点网格
print("X:")
print(X)
print("Y:")
print(Y)

# 绘制生成的二维坐标点网格
plt.scatter(X, Y, color='red')
plt.show()


