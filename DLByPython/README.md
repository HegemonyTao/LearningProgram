# 人工神经网络与卷积神经网络

# 神经网络

### 背景

**感知机**

感知机接收多个输入信号，输出一个信号。如下图所示即为一个感知机

![1-1](./Image/1-1.jpg)

<img src="http://latex.codecogs.com/gif.latex?y=\begin{cases}0\quad (w_1x_1+w_2\le \theta)\\1\quad (w_1x_1+_2x_2>\theta)\end{cases}"/>

常见的应用有**与门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(0.5,0.5,0.7)"/>），**或门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(0.5,0.5,-0.2)"/>）以及**与非门**（<img src="http://latex.codecogs.com/gif.latex?(w_1,w_2,\theta)=(-0.5,-0.5,-0.7)"/>）

但一般的感知机只能解决线性问题，无法解决如异或门这样的问题。可以使用**多层感知机**组合多个门电路来实现，如下所示：

![1-2](./Image/1-2.jpg)