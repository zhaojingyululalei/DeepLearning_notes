import numpy as np
import matplotlib.pyplot as plt
import draw_math_graph as dmg

#制作数据
def make_data(k,k_num,k_center):
    samples = list()
    for i in range(k):
        num = k_num[i]
        center = k_center[i]
        sample = np.random.randn(num,2) + np.array(center)
        samples.append(sample)
    return samples


if __name__ == '__main__':
    k = 3
    k_num = [30,30,30]
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
    
    plt.show()