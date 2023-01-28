import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import *
class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss
net=simpleNet()
print(net.W)
x=np.array([0.6,0.9])
p=net.predict(x)
t=np.array([0,0,1])
print(net.loss(x,t))
def f(W):
    return net.loss(x,t)
print(net.W.size)
dW=numerical_gradient(f,net.W)
print(dW)
