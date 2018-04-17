download: [https://arxiv.org/pdf/1511.06085.pdf]

title: Variable Rate Image Compression with Recurrent Neural Networks

authors: George Toderici, Sean M. O'Malley, Sung Jin Hwang, Damien Vincent, David Minnen, Shumeet Baluja, Michele Covell, Rahul Sukthankar

[TOC]

## 0. Abstract


互联网上大部分流量都是由手机设备所驱动的，并且对带宽要求很严格。因此对现代 重图像 的网站来说，传输低分辨率，低字节数的图像预览（缩略图）作为初始页面加载时后的部分以提高页面的响应能力是很普遍的。
提出了一个用于可变码率的图像压缩的通用框架，还有一个全新的基于卷积和反卷积的LSTM递归神经网络的架构。

解决了阻止auto-encoder与其他已经存在的图像压缩算法相媲美的主要问题:

1. 我们的网络只需要训练一次(而不是每张图片一次), 不论输入图像的尺寸和所希望的压缩率;
2. 我们的网络是逐步求精式的，意味着越多的bits被发送，重建的图像就会越精确;
3. 所提出的框架在给定码元数量下，至少和标准的专门训练的auto-encoder的效率是差不多的。

在大规模的32x32缩略图下，我们的基于LSTM的方法比(不带头信息的)JPEG, JPEG-2000和WebP的视觉质量都要好，同时存储尺寸也减少了10%或更多.

## 1. Introduction

JPEG,JPEG-2k (Joint Pictures Experts Group) 
WebP (Google, 2015)

所有这些方法都来自一个经验性的立场点:  人类专家设计各种启发式规则来减少需要保留的信息量，然后将其进行无损压缩。 

但是研究的主要都是在压缩大图片，低分辨率图像就经常被忽略(甚至会因为大的文件头信息而受到损害)。

缩略图比大图可能包含的高频信息更多，更难压缩。

用神经网络去压缩，未曾认真被人工设计过的图像尺寸，很有意义。

为了满足这种灵活性，我们讨论的网络架构必须满足以下要求:

1. 压缩率必须能受限于之前指定的值
2. 压缩器能够对简单的区域用更少的比特编码，对复杂视觉特征用更多的比特
3. 模型能够从大量已经存在的图像中学习，以优化该压缩来适应真实世界的数据




## 2. Related Work

feed-forward neural network  1999



auto-encoder 是一种可行的实现end-to-end图像压缩的手段 ， 一般分为3个部分:

1. encoder:  消耗输入(一个固定维度的图像或Patch)， 并将其进行转换;
2. bottleneck: 表示压缩后的数据， 随后被转换;
3. decoder: 将其转换为像原始输入的图像.

bottleneck通常是简单的neural net layer，它允许encoded images的压缩率和视觉保真被控制，通过调节该层的node的数量. 对某些类别的autoencoder来说，将bottleneck编码为一个简单的bit vector可能会更有益。


LSTM


## 3. Variable Rate Compression Architectures

x' = D(B(E(x)))

### 3.1 Image Compression Framework


解码应该是progressive的才行

我们chain起来多个residual autoencoder Ft, 定义如下:

![_2018-04-17_20-10-02.png](https://a.photo/images/2018/04/17/_2018-04-17_20-10-02.png)

r_0 为 original input patch
r_t 代表 经过t stages后的residual error.

对于non-LSTM architectures, Ft 没有记忆, 所以我们只希望它能预测residual本身.
每一个stage的惩罚都是根据预测和前面residual之间的差:

![_2018-04-17_20-16-02.png](https://a.photo/images/2018/04/17/_2018-04-17_20-16-02.png)


对于LSTM-based architecture，其做了hold state, 因此我们期望它们在每一stage里面都能预测原始图像patch，对应的，我们计算residual相对于original patch:

![_2018-04-17_20-22-06.png](https://a.photo/images/2018/04/17/_2018-04-17_20-22-06.png)

在这两种情况下，the full, multi-stage network被训练通过优化 minimizing |rt|^2 for t = 1, ... N, 其中N是residual auto-encoders的总数量.


### 3.2 Binary Representation

binarization has 3 benefits:

1.  bit vectors are trivially serializable/deserializable for image transmission over the wire,
2.  control of the network compression rate is achieved simply by putting constraints on the bit allowance,
3.  a binary bottleneck helps force the network to learn efficient representations compared to standard floating-point layers, which may have many redundant bit patterns that have no effect on the output.

Binarization process 包含2部分:

第一部分 产生需要数量的输出(等于所期望的bits), 它在连续区间[-1,1];
第二部分 将这个连续的表示作为输入，产生离散的输出值 {-1,1}.

第一步我们用全连接层，带有tanh激活函数;
第二步, 一种可能的binarization b(x) of x ∈ [-1,1] 定义为:

![_2018-04-17_20-38-43.png](https://a.photo/images/2018/04/17/_2018-04-17_20-38-43.png)

概率是均匀变化的。。

其中 e 对应于量化noise, 我们使用regularization provided by the randomized quantization to allow us to cleanly backpropagate gradients through this binarization layer.

反向传播， E(b(x)) = x,  所以 梯度直接传过去，不改变。
因此，完整的binary encoder function is:

![_2018-04-17_20-47-40.png](https://a.photo/images/2018/04/17/_2018-04-17_20-47-40.png)


为了让某个确定的输入,有固定的representation,一旦网络训练好了, 只有b(x)最可能的输出才被考虑，因此b被替换为b^inf，定义如下:

![_2018-04-17_20-55-26.png](https://a.photo/images/2018/04/17/_2018-04-17_20-55-26.png)

### 3.3 Feed-forward fully-connected residual encoder


### 3.4 LSTM-based compression

E and D consist of stacked LSTM layers

上标指示layer number
下标指示time steps

LSTM架构可以简洁地表示为如下：

![_2018-04-17_21-10-15.png](https://a.photo/images/2018/04/17/_2018-04-17_21-10-15.png)

其中sigm代表 sigmoid function.

### 3.5 Feed-forward convolutional/deconvolutional residual encoder

the final layer of decoder consist of a 1x1 convolution with 3 filters that convert the decoded representation into RGB values.


deconv operator is defined as the transpose of the conv operator.

### 3.6 Conv/Deconv LSTM compression

将T转变为卷积+bias

![_2018-04-17_21-55-21.png](https://a.photo/images/2018/04/17/_2018-04-17_21-55-21.png)

注意: 第二个卷积项代表convolutional LSTM的递归关系, 所以它的输入和输出必须有相同的尺寸，因此stride只作用到第一个卷积项上，而第二个卷积项总是stride=1.

### 3.7 Dynamic Bit Assignment
对于同一个patch, 很自然可以动态分配bits的数量，通过不同的number of iteration of encoder, 它可以用一个目标质量指标(如psnr)来决定。虽然不自然,对于卷积网络，类似的方法可以被使用，如输入图像需要被拆分为patches, 每个patch独立处理，因此就允许不同数量的bits每个region. 然而，这样的方法有缺点，后续会讲.


## 4. Experiments & Analysis

### 4.1 Training
### 4.2 Evaluation protocol and metrics
### 4.3 32x32 benchmark
### 4.4 Analysis

## 5. Conclusion & Future work
...




