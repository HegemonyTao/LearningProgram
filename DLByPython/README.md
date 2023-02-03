# 神经网络和卷积神经网络

## 神经网络前言

### 感知机

感知机接收多个输入信号，输出一个信号

![1-1](./Image/1-1.jpg)

据此可以设计出[与门](Perceptron.py)、[或门](Perceptron.py)以及[与非门](Perceptron.py)；但感知机无法解决异或问题，可使用如下[多层感知机](Perceptron.py)

![1-1](./Image/1-2.jpg)

### 激活函数

决定如何来激活输入信号的总和

#### sigmoid函数

$h(x)=\frac{1}{1+exp(-x)}$				 [实现](./common/functions.py)

#### 阶跃函数

$h(x)=\begin{cases}1,x>0\\0.x\le0\end{cases}$				[实现](./common/functions.py)

#### ReLU函数

$h(x)=\begin{cases}x\quad(x>0)\\0\quad(x\le0)\end{cases}$		  [实现](./common/functions.py)

### 输出层

回归任务使用[恒等函数](./common/functions.py)$h(x)=x$，分类任务常使用[softmax](./common/functions.py)

[常规softmax](./common/functions.py)：$y_k=\frac{exp(a_k)}{\sum_{i=1}^n exp(a_i)}$，会出现溢出问题

[改进softmax](./common/functions.py)：$y_k=\frac{exp(a_k+log C^{'})}{\sum_{i=1}^n exp(a_i+C^{'})}$，$C^{'}$一般会使用输入信号的最大值

### 损失函数

用于评价网络的优劣，常使用均方误差或交叉熵损失

#### 均方误差



