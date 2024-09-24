import numpy as np

#layer [1,10,1] 三层，每层神经元个数
def create_network(layer):
    network = dict()  #字典，每个项存矩阵
    for i in range(1,len(layer)):
        network["w"+str(i)] = np.random.random(size=(layer[i],layer[i-1]))
        network["b"+str(i)] = np.random.random(size=(layer[i],1))
    
    return network

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(network,X):
    z = list()
    #a是z经过sigmod激活的值
    a = list()
    z.append(X)
    a.append(X)

    layer = len(network) 
    for i in range(1,layer - 1):
        w = network["w"+str(i)]
        b = network["w"+str(i)]

        res = np.dot(w,a[i-1]) + b
        z.append(res)
        #最右一层不用sigmod激活
        if i<layer:
            res = sigmoid(res)
        a.append(res)

    return z,a
def sigmod_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))
def backward(network,z,a,y):
    layer = len(network) -1
    grades = dict()
    delta = dict()
    delta[layer] = a[layer] - y
    grades["w"+str(layer)] = np.dot(delta[layer],a[layer-1].T)
    grades["b"+str(layer)] = np.array(delta[layer])
    for i in range(layer,0,-1):
        WT = network["w"+str(layer)].T
        delta[i] = np.dot(WT,delta[i+1])*sigmod_gradient(z[i])

        grades["w"+str(i)] = np.dot(delta[i],a[i-1].T)
        grades["b"+str[i]] = np.array(delta[layer])
    return grades



if __name__ == '__main__':
    network = create_network([2,3,2])
    print(network["b2"].shape)
    X=np.array([
        [1.0,2.0],
        [2.0,1.0],
        [1.5,2.1],
        [0.3,0.6]
    ])
    z,a = forward(network,X.T)