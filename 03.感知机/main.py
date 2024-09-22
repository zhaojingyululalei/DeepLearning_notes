import draw_math_graph as dmg
import numpy as np
import matplotlib.pyplot as plt

def plot_samples_and_boundry(X,Y,w,b,title):
    dmg.draw_coordinate_system(2,-2,-2,2,
                               title,"x","y")
    #把所有样本点打印出来
    for x,y in zip(X,Y):
        if y==0:
            color = "r"
            marker = "x"
        else:
            color = "b"
            marker = "o"
        plt.scatter(x[0],x[1],color=color,marker=marker)
    #把此时的边界画出来
    x1 = np.linspace(-2,2,100)
    x2 = (-1 * w[0] * x1 - b) / w[1]
    plt.plot(x1,x2,"g")
    plt.show()

def predict(x,w,b):
    return np.where(np.dot(w,x)+b > 0,1,0)

def update(x,y,w,b,eta):
    o = predict(x,w,b)
    w = w + eta * (y-o) * x
    b = b + eta * (y-o)
    return w,b


w = np.array([1.0,-1.0])
b = 0.5
eta = 0.3
X = np.array([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
])
Y = np.array([0,0,0,1])
iteration = 0
plot_samples_and_boundry(X,Y,w,b,"0 epoch")

errors = True
while errors!=0:
    errors = 0
    iteration+=1
    for x,y in zip(X,Y):
        if predict(x,w,b) != y:
            w,b = update(x,y,w,b,eta)
            errors += 1
    title = "%d epoch" %(iteration)
    plot_samples_and_boundry(X,Y,w,b,title)



    