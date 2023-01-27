import numpy as np
#简单与门
def ANDS(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    output=w1*x1+w2*x2
    if output>theta:
        return 1
    return 0
#复杂与门：引入权重和偏置
def ANDC(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    output=np.sum(w*x)+b
    if output>0:
        return 1
    return 0
#与非门：引入权重和偏置
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    output=np.sum(w*x)+b
    if output>0:
        return 1
    return 0
#或门
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    output=np.sum(w*x)+b
    if output>0:
        return 1
    return 0
#组合与门、或门和与非门实现异或
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=ANDC(s1,s2)
    return y
