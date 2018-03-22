download:   [https://arxiv.org/pdf/1801.04260.pdf]

title:  Conditional Probability Models for Deep Image Compression

authors:   Fabian Mentzer∗ Eirikur Agustsson∗  Michael Tschannen Radu Timofte Luc Van Gool   ETH Zurich, Switzerland


## Abstract

基于DNN来做图像压缩的挑战主要是两个：
1. 如何处理量化
2. 如何控制distortion和rate的trade-off
本文主要关注第2个问题。
即 优化d+βR，其中d是reconstruction的distortion，R是bitrate，信息论告诉我们R是编码的熵H，如何才能建模H，来平衡auto-encoder的d+βH的trade-off.

本文提出的方法是利用一个context model，直接在优化时作为一个entropy term，并行同时训练auto-encoder和context model，后来又扩展方法到spatial-aware，即添加了importance map去指导空间bitrate的分配。

## Related Work

DNN压缩图像的方法一般有两种:
1. auto-encoders
2. RNNs

关于如何利用context models来进行entropy estimation，有几种不同的实现方法：
1. 有的利用一个binary context model来进行动态binary算术编码
2. 有的利用一个learned context model来提高编码的码率（如Li.）
3. 有的利用非动态算术编码，但是用symbols上一个独立假设来估计entropy term.

PixelRNN和PixelCNN也是基于RNN和CNN的context-model，可以用来建模自然图像的分布，用于无损图像压缩（adaptive arithmetic coding），也可以用于图像生成（抽样），（本质上，就是建模了自然图像的分布）。

## 提出的方法

给定自然图像的集合X，希望学到encoderE-quantizerQ-decoderD。

E：R^d ==> R^m  映射图像x到latent representation z=E(x)

Q：R ==> C  离散化z的每维到L=|C|个中心，得到z^，之后可以被无损压缩到二进制流

D: x^ = D(z^)

rate-distortion tradeoff:

![9HTjAK.png](https://s1.ax1x.com/2018/03/22/9HTjAK.png)


H代表把z^编码到bits的代价，也就是entropy of z^

### 3.1 量化

用最近邻来量化

![9HTvtO.png](https://s1.ax1x.com/2018/03/22/9HTvtO.png)

我们给定C={c1, ... , cL} < R

反向传播时，使用soft quantization(可微分)来计算梯度：

![9HTxhD.png](https://s1.ax1x.com/2018/03/22/9HTxhD.png)

C是学习的。

`tf.stopgradient(zi^ - zi~) + zi~`


### 3.2 Entropy Estimation

将z^表示为条件概率的乘积。

![9H71H0.png](https://s1.ax1x.com/2018/03/22/9H71H0.png)

raster scan order 将z^这个三维feature volume转变为了一维。

用一个神经网络P(z^)，即context model，来估计每一项

![9H70D1.png](https://s1.ax1x.com/2018/03/22/9H70D1.png)

这可以并行进行，类似于Trimmed CNN.

由MLE，等价的，可以用Pi:来预测symbol，用交叉熵做损失：

![9HHxQH.png](https://s1.ax1x.com/2018/03/22/9HHxQH.png)

将p(z^)替换为近似的q(z^)，我们可以将CE Loss看成H(z^)的一个近似，因为我们学习到的P=q≈p，

![9Hbumn.png](https://s1.ax1x.com/2018/03/22/9Hbumn.png)

将CE的期望内的表达式叫做coding cost
[![9HbtX9.png](https://s1.ax1x.com/2018/03/22/9HbtX9.png)](https://imgchr.com/i/9HbtX9)

### 3.3 并行优化

给定auto-encoder (E,D)，训练P来model z^中各项的依赖关系，同时P的CE Loss又是Entropy的估计。

与GAN的训练很类似，GAN中的G和D都是同时在被优化，但是两者优化各自的adversarial loss，但与GAN不同的是，我们的两个目标不是直接对抗的，如果P预测的更准，则H(z)就更准，从而帮助(E,D)来更好的优化rate-distortion。


### 3.4 Importance map for spatial bit-allocation

理论上，是可以自动学到spatial bit-allocation的，但是这需要更加复杂的网络，直接用一个importance map结构会更有效。

输入图片如果是(W,H)的，则z^的维度将会是(W/8, H/8, K)。

[![9HqtC8.png](https://s1.ax1x.com/2018/03/22/9HqtC8.png)](https://imgchr.com/i/9HqtC8)

yij并不是(0,1)，然后m还要ceil(m)
z <- z .* ceil(m)， 