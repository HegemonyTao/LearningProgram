# 自然语言处理综述

NLP模型根据其主要架构可分为三类：RNNs，CNNs和attention-based

## 循环神经网络（RNN）

传统RNN存在误差减小问题，有用**ReLU**代替sigmoid函数的、使用**LSTM**的，或者使用门控循环网络(**GRUs**)的

但这些架构计算速度慢，在确定当前状态时无法考虑未来的输入，以及访问过去信息的额外挑战。最重要的是无法并行

## 编码器-解码器架构

Seq2Seq通常使用这种架构，如下图所示

![encoder-decoder](./Image/Review/encoder-decoder.jpg)

但这种架构存在两个问题：

* 编码后的数据经过压缩形成一个固定长度的向量，再将这个向量发送到解码器，这个过程中会有丢失信息的风险
* 无法描述需要结构化输出的任务(如翻译和摘要)所必须的输入-输出序列对齐

## Transformer

最初由**Vaswani**提出，用于解决rnn和编码器-解码器的缺点

### 注意力机制

此机制使用解码器状态（Q:query）和编码器隐藏状态（K:keys）来计算注意力权重，在历史上表示编码器隐藏状态（V:values）在处理解码器状态的相关性。一个称为注意力（Q,K,V）的广义注意力模型使用一组键值对(K,V)和一个称为Q的查询：

![Equation-1](./Image/Review/Equation-1.jpg)

其中，$F_{distribution}$是分布函数，常常使用logistic,sigmoid和softmax函数等。$F_{alignment}$是对齐函数，原始Transformer中使用**点积对齐**，其他常见的对齐函数如下：

![AlignmentFunctions](./Image/Review/AlignmentFunctions.jpg)

自注意也称内注意，它将单个序列的各个位置联系起来，以创建序列的表示：

![self-attention](./Image/Review/self-attention.jpg)

**多头注意层**由多个注意头组成。每个注意头计算其输入V,K和Q的注意力，同时使它们进行线性转换，如下所示：

![multi-head-Attention](.\Image\Review\multi-head-Attention.jpg)

掩码多头注意网络参与之前的解码器状态，其中$H^0={x_1,...,x_{|x|}}$，$H^l=Transformer_l(H^{l-1}),l\in [1,L]$，自注意的输出如下所示：

![Equation-2](./Image/Review/Equation-2.jpg)

### 位置编码

 位置编码用于解决注意力机制忽略了关于每个单词位置的细节，可以根据每个单词在当前序列中的位置对其进行编码。并使用如下公式计算：

![Equation-3](./Image/Review/Equation-3.jpg)

其中，$d_{model}$是嵌入维数（ embedding dimension），$pos_{word}$是序列中的位置（0到n-1），$pos_{emb}$是嵌入维数中的位置（从0到$d_{model}-1$）。**Transformer-XL**就使用了相对位置编码

### 位置前馈网络



