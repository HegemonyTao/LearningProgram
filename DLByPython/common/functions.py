import numpy as np
def identity_function(x):
    return x
#阶跃函数实现：支持Numpy数组
def step_function(x):
    return np.array(x>0,dtype=np.int_)
#sigmoid函数实现
def sigmoid(x):
    return 1/(1+np.exp(-x))
#sigmoid函数导数
def sigmoid_grad(x):
    return (1.0-sigmoid(x))*sigmoid(x)
def relu(x):
    return np.maximum(0,x)
#relu函数导数
def relu_grad(x):
    grad=np.zeros(x)
    grad[x>=0]=1
    return grad
def softmax(x):
    #二维情况
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
    x=x-np.max(x)
    return np.exp(x)/np.sum(np.exp(x))
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)
def cross_entropy_error(y,t):
    #一维
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    else:
        if y.shape[0]==t.shape[1]:
            y=y.T
    if t.size==y.size:
        t=t.argmax(axis=1)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
def softmax_loss(X,t):
    y=softmax(X)
    return cross_entropy_error(y,t)