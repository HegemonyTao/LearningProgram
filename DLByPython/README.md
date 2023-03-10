- [神经网络](#神经网络)
  - [感知机](#感知机)
  - [激活函数](#激活函数)
    - [Sigmoid函数](#sigmoid函数)
    - [阶跃函数](#阶跃函数)
    - [ReLU函数](#relu函数)
  - [损失函数](#损失函数)
    - [均方误差](#均方误差)
    - [交叉熵损失](#交叉熵损失)
    - [mini-batch学习](#mini-batch学习)
  - [学习算法](#学习算法)
    - [数值微分和梯度](#数值微分和梯度)
    - [计算图计算梯度](#计算图计算梯度)
      - [加法节点](#加法节点)
      - [乘法节点](#乘法节点)
      - [ReLU层](#relu层)
      - [Sigmoid层](#sigmoid层)
      - [Softmax-with-Loss](#softmax-with-loss)
  - [学习技巧](#学习技巧)
    - [参数更新](#参数更新)
      - [SGD](#sgd)
      - [Momentum](#momentum)
      - [AdaGrad](#adagrad)
      - [Adam](#adam)
    - [权重初值](#权重初值)
    - [Batch Normalization](#batch-normalization)
    - [正则化](#正则化)
      - [权值衰减](#权值衰减)
      - [Dropout](#dropout)
- [人工神经网络的实现](#人工神经网络的实现)
- [卷积神经网络](#卷积神经网络)
  - [整体结构](#整体结构)
  - [卷积层](#卷积层)
    - [卷积运算](#卷积运算)
    - [填充](#填充)
    - [步幅](#步幅)
  - [池化层](#池化层)
  - [实现](#实现)

# 神经网络

## 感知机

感知机接收多个输入信号，输出一个信号。如下图所示即为一个感知机

![1-1](./Image/1-1.jpg)

<img src="http://latex.codecogs.com/gif.latex?y=\begin{cases}0\quad (w_1x_1+w_2\le \theta)\\1\quad (w_1x_1+_2x_2>\theta)\end{cases}"/>

常见的应用有**与门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(0.5,0.5,0.7)"/>），**或门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(0.5,0.5,-0.2)"/>）以及**与非门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(-0.5,-0.5,-0.7)"/>）

但一般的感知机只能解决线性问题，无法解决如异或门这样的问题。可以使用**多层感知机**组合多个门电路来实现，如下所示：

![1-2](./Image/1-2.jpg)

## 激活函数

激活函数用来决定如何来激活输入信号的总和，常见的激活函数如下

### Sigmoid函数

<img src="http://latex.codecogs.com/gif.latex?h(x)=\frac{1}{1+exp(-x)}"/>

### 阶跃函数

<img src="http://latex.codecogs.com/gif.latex?h(x)=\begin{cases}0\quad (x\le0)\\1\quad(x>0)\end{cases}"/>

### ReLU函数

<img src="http://latex.codecogs.com/gif.latex?h(x)=\begin{cases}x\quad(x>0)\\0\quad(x\le0)\end{cases}"/>

## 损失函数

神经网络的学习中所用的指标称为损失函数，它是表示神经网络性能的“恶劣程度”的指标，通常使用均方误差和交叉熵损失误差等。

### 均方误差

<img src="http://latex.codecogs.com/gif.latex?E=\frac{1}{2}\sum_k(y_k-t_k)^2"/>

其中，<img  src="http://latex.codecogs.com/gif.latex?y_k"/>表示神经网络的输出，<img src="http://latex.codecogs.com/gif.latex?t_k"/>表示监督数据，k表示数据的维度

### 交叉熵损失

<img src="http://latex.codecogs.com/gif.latex?E=-\sum_kt_klog(y_k)"/>

其中，log表示以e为底的自然对数，<img src="http://latex.codecogs.com/gif.latex?y_k"/>是神经网络的输出，<img src="http://latex.codecogs.com/gif.latex?t_k"/>是正确解标签

### mini-batch学习

如果使用mini-batch来学习，损失函数应该做出相应的改变，如交叉熵损失函数应变为如下形式：

<img src="http://latex.codecogs.com/gif.latex?E=-\frac{1}{N}\sum_n\sum_kt_{nk}log(y_{nk})"/>

其中，数据有N个，<img src="http://latex.codecogs.com/gif.latex?t_{nk}"/>表示第n个数据的第k个元素的值（<img src="http://latex.codecogs.com/gif.latex?y_{nk}"/>是神经网络的输出，<img src="http://latex.codecogs.com/gif.latex?t_{nk}"/>是监督数据）

## 学习算法

### 数值微分和梯度

对于函数导数的近似求解，可采用数值微分的方法，即：

<img src="http://latex.codecogs.com/gif.latex?\frac{df(x)}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}"/>

对于偏导数，固定一个变量的值，求出另一个变量的近似导数即可。在更新时，常使用梯度下降法来做，如：

<img src="http://latex.codecogs.com/gif.latex?x_0=x_0-\eta\frac{d f}{d x_0}\\x_1=x_1-\eta\frac{d f}{d x_1}"/>      

### 计算图计算梯度

计算图可以通过局部计算，将复杂的全局计算转化为简单的局部计算。具体方式是将复杂图拆分为一个一个简单的节点，而后使用链式法则进行求解即可。

#### 加法节点

对于z=x+y，先z对x及y分别求导，可得如下偏导数：

<img src="http://latex.codecogs.com/gif.latex?\frac{d z}{d x}=1\\ \frac{d z}{d y}=1"/>

其反向传播可以通过下图表示：

![AddNodeCG](./Image/AddNodeCG.jpg)

#### 乘法节点

考虑z=xy，可以得到如下偏导数：

<img src="http://latex.codecogs.com/gif.latex?\frac{d z}{d x}=y\\ \frac{d z}{d y}=x"/>

其反向传播可以通过下图表示：

![MultiNodeCG](./Image/MultiNodeCG.jpg)

#### ReLU层

ReLU原函数如下：

<img src="http://latex.codecogs.com/gif.latex?y=\begin{cases}x\quad(x>0)\\0\quad(x\le0)\end{cases}"/>

求出y关于x的导数如下：

<img src="http://latex.codecogs.com/gif.latex?\frac{d y}{d x}=\begin{cases}1\quad(x>0)\\ 0\quad(x\le0)\end{cases}"/>

其计算图如下：

![ReLUNodeCG](./Image/ReLUNodeCG.jpg)

#### Sigmoid层

Sigmoid原函数：

<img src="http://latex.codecogs.com/gif.latex?y=\frac{1}{1+exp(-x)}"/>

Sigmoid导数：

<img src="http://latex.codecogs.com/gif.latex?\frac{dy}{dx}=y(1-y)"/>

其计算图如下：

![SigmoidNodeCG](./Image/SigmoidNodeCG.jpg)

#### Softmax-with-Loss

![SoftmaxLossNodeCG](./Image/SoftmaxLossNodeCG.jpg)

## 学习技巧

### 参数更新

#### SGD

![SGDFormula](./Image/SGDFormula.jpg)

但如果函数的形状非均向，搜索的路径就会非常低效，如下图所示：

![SGDOptimization](./Image/SGDOptimization.jpg)

#### Momentum

![MomentunFormula](./Image/MomentunFormula.jpg)

![MomentumOptimization](./Image/MomentumOptimization.jpg)

#### AdaGrad

![AdaGradFormula](./Image/AdaGradFormula.jpg)

![AdaGradOptimization](./Image/AdaGradOptimization.jpg)

#### Adam

融合了Momentum和AdaGrad的方法

![AdamOptimization](./Image/AdamOptimization.jpg)

### 权重初值

* 权重的初始值一般不要设为0，可以尝试使用均值小的正态分布

* 可以尝试使用Xavier初始值等来初始化权重

### Batch Normalization

为了使得各层拥有适度的广度，从而可以顺利地进行学习，一般是插入在激活函数层前面。它以进行学习时的mini-batch为单位，按mini-batch进行正规化，即进行均值为0，方差为1的正则化，如下所示：

<img src="http://latex.codecogs.com/gif.latex?\mu_B\leftarrow\frac{1}{m}\sum_{i=1}^mx_i\\\sigma_B^2\leftarrow\frac{1}{m}\sum_{i=1}^m(x_i-\mu_B)^2\\\hat{x_i}\leftarrow\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\xi}}\\y_i\leftarrow\gamma\hat{x_i}+\beta"/>

![BatchNormalizationNodeCG](./Image/BatchNormalizationNodeCG.jpg)

### 正则化

在网络的训练过程中很容易出现过拟合。一般来说，是由于模型拥有大量参数、表现力强或者是训练数据少造成的。可使用如下方式来抑制过拟合：

#### 权值衰减

该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。具体方式就是为损失函数加上权重的平方范数（L2范数），从而抑制权重变大。

#### Dropout

当网络变得复杂时，就难以只使用权值衰减来解决了，此时可采用Dropout方法。如下所示：

![Dropout](./Image/Dropout.jpg)

# 人工神经网络的实现

**三层神经网络进行手写数字体(MNIST)识别**（使用激活函数、Batch-Normalization、Dropout、误差反向传播算法，Softmax-with-Loss等）

# 卷积神经网络

## 整体结构

![CNNFullStructure](./Image/CNNFullStructure.jpg)

其中，Conv是卷积层，Pooling是池化层

## 卷积层

### 卷积运算

使用滤波器，在输入数据的对应位置进行乘法并求和，而后也可以加上偏置，如下图所示：

![ConvOperation](./Image/ConvOperation.jpg)

### 填充

在进行卷积层的处理之前，有时需要向输入数据的周围填固定的数据，这称为填充，如下图所示：

![PaddingOperation](./Image/PaddingOperation.jpg)

### 步幅

应用滤波器的位置间隔称为步幅，如下所示：

![StrideOperation](./Image/StrideOperation.jpg)

假如输入大小为(H,W)，滤波器大小为(FH,FW)，输出大小为(OH,OW)，填充为P，步幅为S。此时，输出大小可通过如下方式进行计算：

<img src="http://latex.codecogs.com/gif.latex?OH=\frac{H+2P-FH}{S}+1\\OW=\frac{W+2P-FW}{S}+1"/>

## 池化层

池化层是缩小高、长方向上的空间的运算。常见的池化有：Max池化、Average池化等，如下是Max池化例子：

![MaxPoolingOperation](./Image/MaxPoolingOperation.jpg)

## 实现

使用CNN进行手写数字体识别

**具有代表性的CNN**

LeNet：激活函数使用sigmoid函数，使用子采样缩小中间数据的大小

![LeNet](./Image/LeNet.jpg)

AlexNet：激活函数使用ReLU、使用进行局部正规化的LRN以及使用Dropout

![AlexNet](./Image/AlexNet.jpg)
