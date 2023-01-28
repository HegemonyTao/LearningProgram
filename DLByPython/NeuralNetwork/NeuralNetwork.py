import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import cross_entropy_error,sigmoid,identity_function
from common.gradient import numerical_gradient
import matplotlib.pylab as plt
#三层神经网络实现
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network
#前向传播算法
def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    hidden1I=np.dot(x,W1)+b1
    hidden1O=sigmoid(hidden1I)
    hidden2I=np.dot(hidden1O,W2)
    hidden2O=sigmoid(hidden2I)
    outputI=np.dot(hidden2O,W3)+b3
    y=identity_function(outputI)
    return y
#梯度下降法
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x

def function_2(x):
    return x[0]**2+x[1]**2
init_x=np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))

network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)

