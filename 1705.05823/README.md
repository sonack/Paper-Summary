download:   [https://arxiv.org/pdf/1705.05823.pdf]

title:  Real-Time Adaptive Image Compression

authors: Oren Rippel * 1 Lubomir Bourdev *  [equal contribution  WaveOne Inc.]

[TOC]

## 0. Abstract


files 2.5x smaller than JPEG and JPEG 2000,
2x smaller than WebP,
1.7x smaller than BPG.

speed:  encode or decode Kodak dataset in around 10ms per image on GPU.



architecture: autoencoder featuring pyramidal(金字塔) analysis, an adaptive coding module, and regularization of the expected codelength(控制码率)

此外，我们也用GAN来补足我们的方法，以便在很低的码率下也能够产生视觉效果还可以的重建。



## 1. Introduction

数字流媒体的流量占据了互联网的70%, 预计到2020将占据80%. 现在的codec都是"one-size-fits-all".

alleviate 减轻
深度学习为何还没有充分改变压缩这个经典领域呢? 主要是两个原因:
1. 深度学习的基元，在它们的原始形式，本来就不适合产生足够紧凑的表示
2. 发展一个深度学习方法，足够有效率，可以用来部署在计算能力、内存、电池受限的环境下还很困难。


在GTX 980 Ti GPU, 花费大约9ms to encode and 10ms to decode an image , 对于JPEG, encode/decode time is 18ms/12ms, 对于JP2, 350ms/80ms, for WebP 70ms/80ms.


## 2. Background && Related Work

### 2.1 Traditional compression techniques

compression, 和 pattern recognition 其实关系很紧密， 如果我们能发掘输入的结构， 那么我们就能消除冗余信息，用更简洁的表示。

JPEG, 8x8 block, DCT transforms, run-length encoding
JP2 employs an adaptive arithmetic coder 去捕获之前multi-resolution wavelet transform产生的系数大小的分布。

### 2.2 ML-based lossy image compression

auto-encoder 

### 2.3 Generative Adversarial Networks

![_2018-04-18_20-11-53.png](https://a.photo/images/2018/04/18/_2018-04-18_20-11-53.png)

## 3. Model


![_2018-04-18_20-18-48.png](https://a.photo/images/2018/04/18/_2018-04-18_20-18-48.png)

**Feature Extraction**

images feature包括很多不同类型的结构: 

1. input channels间的
2. 单个尺度内的
3. 跨尺度的
等等

我们设计的特征提取器架构，包含了一个分析单个尺度的pyramidal decomposition, 后面是一个跨尺度的对齐过程，其充分利用跨尺度的共享结构.

**Code computation and regularization**

用来进一步压缩抽取的特征, 它量化了特征， 然后通过adaptive arithmetic coding作用到它们的二进制展开上。一个adaptive codelength regularization被引入来惩罚features的entropy, 它可以被充分利用以便达到更好的压缩性能。

**Discriminator loss**

pursue 追赶, 继续, 纠缠

realistic reconstruction

### 3.1 Feature extraction

#### 3.1.1 Pyramidal decomposition

类比于JP2中的multi-resolution wavelets analysis,
learn optimal, nonlinear extractor individually for each scale

假设模型的输入为x, 一共有M个scales.

recursive analysis:

假设尺度m的输入表示为xm, 我们设置第一个scale x1=x, 对于每个scale m, 我们进行2项操作:

首先, 抽取系数 cm = fm(xm) ∈ R(CmxHmxWm) 通过某个参数化的函数fm(*), 输出的通道数为Cm, 高度是Hm, 宽度是Wm.

然后，计算下一个scale的输入，用x(m+1) = Dm(xm), Dm是某个下采样算子(可以是固定的，也可以是学习的)

![_2018-04-18_20-59-40.png](https://a.photo/images/2018/04/18/_2018-04-18_20-59-40.png)


M=6, 对于单个尺度的feature extractors由一系列的卷积组成，卷积核大小为3x3或者1x1, and ReLU with a leak of 0.2.

learn all downsampler as 4x4 convolution with a stride of 2.


#### 3.1.2 Interscale alignment

leverage information shared across different scales

输入为 {cm} m=1..M, 其尺寸为R(CmxHmxWm)
输出 为 一个单独的tensor, 其尺寸为CxHxW.


为此，我们首先将每一个输入tensor cm 映射到目标尺寸CxHxW，通过参数化的函数gm(*),然后再加起来，最后再来个gm()带参非线性变换。


如上图右半部分，输出为y, 实践中，我们选择gm为卷积或反卷积with a appropriate stride来产生目标空间图尺寸HxW, gm一般选择为简单的3x3卷积的序列.



### 3.2 Code computation and regularization

extracted tensor y ∈ R(CxHxW), 然后是量化和encode它。

量化: 




