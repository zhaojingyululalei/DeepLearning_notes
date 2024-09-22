from mpl_toolkits import axisartist
import matplotlib.pyplot as plt 
import numpy as np
import sympy as sp

#自定义坐标系
def draw_coordinate_system(up,down,left,right,title,xlabel,ylabel):
    board = plt.figure(figsize=(12,8)) #创建画板
    axis = axisartist.Subplot(board,111) #创建坐标系
    board.add_axes(axis) #将坐标系加入画板
    axis.set_aspect("equal") #设置坐标轴纵横比为1:1，一个单位的 x 方向长度等于一个单位的 y 方向长度
    axis.axis[:].set_visible(False)

    axis.axis["x"] = axis.new_floating_axis(0,0)
    axis.axis["x"].set_axisline_style("->")
    axis.axis["x"].set_axis_direction("right")
    axis.set_xlim(left,right)

    axis.axis["y"] = axis.new_floating_axis(1,0)
    axis.axis["y"].set_axisline_style("->")
    axis.axis["y"].set_axis_direction("top")
    axis.set_ylim(down,up)

    # 设置 x y 轴刻度间隔为 1
    axis.set_xticks(range(left, right + 1))
    axis.set_yticks(range(down, up + 1))

    #设置名称，坐标轴名称
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.grid(True)

#绘制直线函数
def draw_liner_function(k,b,left,right):
    x = np.linspace(left,right,100)
    y = k * x + b
    plt.plot(x,y)

#绘制一元二次函数
def draw_quadratic_function(a,b,c,left,right):
     x = np.linspace(left,right,100)
     y = a*x*x + b*x + c
     plt.plot(x,y)

#绘制幂函数
def draw_power_function(a,left,right):
    x = np.linspace(left,right,100)
    y = x ** a
    plt.plot(x,y)

#绘制指数函数
def draw_exponential_function(a,left,right):    
    x = np.linspace(left,right,100)
    y = a ** x
    plt.plot(x,y)

#绘制对数函数,以a为厎
def draw_log_function(a,left,right):
    x = np.linspace(left,right,100)
    y = np.log(x) / np.log(a)
    plt.plot(x,y)

if __name__ == "__main__":
    left = -8
    right = 18
    up = 7
    down = -8
    draw_coordinate_system(up,down,left,right,"test","test","test")
    draw_log_function(3,left,right)
    draw_liner_function(1,-5,left,right)
    plt.legend()
    plt.show()



    

    