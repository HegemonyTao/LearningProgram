# 人工神经网络与卷积神经网络

# 神经网络

### 感知机

感知机接收多个输入信号，输出一个信号。如下图所示即为一个感知机

![1-1](./Image/1-1.jpg)

<img src="http://latex.codecogs.com/gif.latex?y=\begin{cases}0\quad (w_1x_1+w_2\le \theta)\\1\quad (w_1x_1+_2x_2>\theta)\end{cases}"/>

常见的应用有**与门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(0.5,0.5,0.7)"/>），**或门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(0.5,0.5,-0.2)"/>）以及**与非门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(-0.5,-0.5,-0.7)"/>）

但一般的感知机只能解决线性问题，无法解决如异或门这样的问题。可以使用**多层感知机**组合多个门电路来实现，如下所示：

![1-2](./Image/1-2.jpg)

### 激活函数

激活函数用来决定如何来激活输入信号的总和，常见的激活函数如下

#### Sigmoid函数

<img src="http://latex.codecogs.com/gif.latex?h(x)=\frac{1}{1+exp(-x)}"/>

#### 阶跃函数

<img src="http://latex.codecogs.com/gif.latex?h(x)=\begin{cases}0\quad (x\le0)\\1\quad(x>0)\end{cases}"/>

#### ReLU函数

<img src="http://latex.codecogs.com/gif.latex?h(x)=\begin{cases}x\quad(x>0)\\0\quad(x\le0)\end{cases}"/>

### 损失函数

神经网络的学习中所用的指标称为损失函数，它是表示神经网络性能的“恶劣程度”的指标，通常使用均方误差和交叉熵损失误差等。

#### 均方误差

<img src="http://latex.codecogs.com/gif.latex?E=\frac{1}{2}\sum_k(y_k-t_k)^2"/>

其中，<img  src="http://latex.codecogs.com/gif.latex?y_k"/>表示神经网络的输出，<img src="http://latex.codecogs.com/gif.latex?t_k"/>表示监督数据，k表示数据的维度

#### 交叉熵损失

<img src="http://latex.codecogs.com/gif.latex?E=-\sum_kt_klog(y_k)"/>

其中，log表示以e为底的自然对数，<img src="http://latex.codecogs.com/gif.latex?y_k"/>是神经网络的输出，<img src="http://latex.codecogs.com/gif.latex?t_k"/>是正确解标签

#### mini-batch学习

如果使用mini-batch来学习，损失函数应该做出相应的改变，如交叉熵损失函数应变为如下形式：

<img src="http://latex.codecogs.com/gif.latex?E=-\frac{1}{N}\sum_n\sum_kt_{nk}log(y_{nk})"/>

其中，数据有N个，<img src="http://latex.codecogs.com/gif.latex?t_{nk}"/>表示第n个数据的第k个元素的值（<img src="http://latex.codecogs.com/gif.latex?y_{nk}"/>是神经网络的输出，<img src="http://latex.codecogs.com/gif.latex?t_{nk}"/>是监督数据）

### 学习算法

#### 数值微分和梯度

对于函数导数的近似求解，可采用数值微分的方法，即：

<img src="http://latex.codecogs.com/gif.latex?\frac{df(x)}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}"/>

对于偏导数，固定一个变量的值，求出另一个变量的近似导数即可。在更新时，常使用梯度下降法来做，如：

<img src="http://latex.codecogs.com/gif.latex?x_0=x_0-\eta\frac{\partial f}{\partial x_0}\quad\quad"/><img src="http://latex.codecogs.com/gif.latex?x_1=x_1-\eta\frac{\partial f}{\partial x_1}"/>      

#### 计算图计算梯度

计算图可以通过局部计算，将复杂的全局计算转化为简单的局部计算。具体方式是将复杂图拆分为一个一个简单的节点，而后使用链式法则进行求解即可。

##### 加法节点

对于z=x+y，先z对x及y分别求导，可得如下偏导数：

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial z}{\partial x}=1\\ \frac{\partial z}{\partial y}=1"/>

其反向传播可以通过下图表示：

![AddNodeCG](./Image/AddNodeCG.jpg)

##### 乘法节点

考虑z=xy，可以得到如下偏导数：

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial z}{\partial x}=y\\ \frac{\partial z}{\partial y}=x"/>

其反向传播可以通过下图表示：

![MultiNodeCG](./Image/MultiNodeCG.jpg)

##### ReLU层

ReLU原函数如下：

<img src="http://latex.codecogs.com/gif.latex?y=\begin{cases}x\quad(x>0)\\0\quad(x\le0)\end{cases}"/>

求出y关于x的导数如下：

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial y}{\partial x}=\begin{cases}1\quad(x>0)\\ 0\quad(x\le0)\end{cases}"/>

其计算图如下：

![ReLUNodeCG](./Image/ReLUNodeCG.jpg)