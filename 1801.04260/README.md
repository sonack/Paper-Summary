download:   [https://arxiv.org/pdf/1801.04260.pdf]

title:  Conditional Probability Models for Deep Image Compression

authors:   Fabian Mentzer∗ Eirikur Agustsson∗  Michael Tschannen Radu Timofte Luc Van Gool   ETH Zurich, Switzerland

[TOC]

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

将p(z\^)替换为近似的q(z^)，我们可以将CE Loss看成H(z^)的一个近似，因为我们学习到的P=q≈p，

![9Hbumn.png](https://s1.ax1x.com/2018/03/22/9Hbumn.png)

将CE的期望内的表达式叫做coding cost
[![9HbtX9.png](https://s1.ax1x.com/2018/03/22/9HbtX9.png)](https://imgchr.com/i/9HbtX9)

### 3.3 并行优化

给定auto-encoder (E,D)，训练P来model z^中各项的依赖关系，同时P的CE Loss又是Entropy的估计。

与GAN的训练很类似，GAN中的G和D都是同时在被优化，但是两者优化各自的adversarial loss，但与GAN不同的是，我们的两个目标不是直接对抗的，如果P预测的更准，则H(z)就更小(熵越小，不确定性越小)，从而帮助(E,D)来更好的优化rate-distortion。



### 3.4 Importance map for spatial bit-allocation

理论上，是可以自动学到spatial bit-allocation的，但是这需要更加复杂的网络，直接用一个importance map结构会更有效。

输入图片如果是(W,H)的，则z\^的维度将会是(W/8, H/8, K)。

[![9HqtC8.png](https://s1.ax1x.com/2018/03/22/9HqtC8.png)](https://imgchr.com/i/9HqtC8)

yij并不是(0,1)，然后m还要ceil(m)
z <- z .* ceil(m)


我们可以通过z\^来恢复[m]，只要数一下z\^[i,j,:]中的顶部有多少个连续的0即可（可能会over-estimate, but no matter），因此[m]就是masked z^\的一个函数，即条件熵H([m]|z\^)=0，
即有下面关系成立：

![9b8nJA.png](https://s1.ax1x.com/2018/03/23/9b8nJA.png)
[条件熵和联合熵的关系，你哈记得吗？]

如果我们认为在优化(E,D)时候H([m])为常数(可能是有大片的1或0相连，熵很低)，则可以间接通过优化H(z|m)来优化H(z).

因为[m]为0的地方，z肯定也为0，所以它的熵为0.剩下的，可以同样利用context model P(i,l)(z^)建模：

![9b8azq.png](https://s1.ax1x.com/2018/03/23/9b8azq.png)

类似于coding cost, 我们把期望内的式子叫做"masked coding cost of z^".

虽然masked coding cost的数量和coding cost几乎一样(除了H([m]))，但masked coding 的好处在于它显式地与[mi]相关联，这样encoder E就有了另外一条明显的路径去控制z\^的熵了。

当训练P的时候，还是采用原来的loss，即并不直接接触到mask m，而是需要在整个masked symbol volumns z^上去学习这种mask后的依赖关系，这样的话，就可以直接采用算术编码了，而不用先编码mask，再编码剩下的symbols了。这很关键，因为这样P的学习到的概率才是真正编码的概率，才能有效地联系起来P和E，两者才能够并行训练。

在实验中，可以观察到，两种coding cost的entropy  losses值几乎收敛到相同的数值，masked coding cost比前者小大约3.5%，由于H([m])被忽略。

固定channel depth K后，通过β控制z\^的熵变得更加容易了，因为网络可以轻易学会利用importance map忽略某些channels。然而，如果K被仔细调整后，imp map并不是优化rate-distortion性能的关键。

### 3.5 Putting the pieces together


pipeline for learned image compression:

给定训练图片集X，初始化(全卷积)CNNs E,D和P，还有量化器Q的centers C.
然后在mini-batches XB = {X1, X2, ... , XB}， 其中Xi都是X中的crops.

我们对auto-encoder (E,D)和quantizer Q采取一步梯度下降，其loss为rate-distortion tradeoff:

![_2018-04-14_13-57-36.png](https://a.photo/images/2018/04/14/_2018-04-14_13-57-36.png)

对context model P采取一步梯度下降，其objective为CE:

![_2018-04-14_14-02-26.png](https://a.photo/images/2018/04/14/_2018-04-14_14-02-26.png)

*?: 第一项的d好像和P没有关系，梯度应该一直为0?*

计算步骤如下:

1. 从encoder中获得压缩表达形式z和重要图y: (z,y) = E(x);
2. Expand imp map y to mask m via mi,j,k = ...;
3. Mask z, 即 z <--  z .* ceil(m);
4. Quantize z\^ = Q(z);
5. Compute the context P(z\^);
6. Decode x\^ = D(z\^);

可以并行，因为全部模型都是全卷积的。


### 3.6 Relationship to previous methods

使用context model来做adaptive arithmetic coding来提高压缩性能的工作之前就有很多，但大多都是在训练了压缩模型后，才单独学习的context model，将其作为了一个提高编码性能的后处理步骤。相反，本文工作直接将context model作为熵项来用作auto-encoder的rate-distortion term，并且并行训练这2个模型。因为采用了3D-CNN来作为context model，因此额外开销很小。

## 4. Experiments

**Architecture**

auto-encoder:

![_2018-04-14_14-19-46.png](https://a.photo/images/2018/04/14/_2018-04-14_14-19-46.png)

* 深灰色块代表residual blocks
* 上半部分是encoder E, 下半部分是decoder D
* 名字的含义: 对于E, k5n64-2代表一个conv层,其kernel size是5, output channel是64, stride是2; 对于D, 它代表对应的deconv层
* 所有的卷积层都使用了BN层, SAME padding
* Masked quantization在3.2中有所描述
* Normalize将输入normalize到[0,1],使用training set的一个子集中计算获得的mean和variance, Denormalize是相反操作.

context model P:
简单的4层3D-CNN
![_2018-04-14_14-21-26.png](https://a.photo/images/2018/04/14/_2018-04-14_14-21-26.png)

* 3Dk3n24代表一个3D masked conv层,其filter size是3,24个output channels
* 最后一层对于z\^中的每个voxel输出L个值

**Distortion measure**

MS-SSIM

d(x, x\^) = 100 * (1 - MS-SSIM(x, x\^)) 

**Training**

| optimizer  |  batch_size | num_of_models| lr| lr_decay_epochs| lr_decay_factor|sigma of Q backward approx|
|---|---|---|---|---|---|---|
| adam  | 30  | 7| 4*10^(-3) |every 2 epochs|10| 1|

每个模型都是直接maximum MS-SSIM.
作为基准，lr=0.004，但是针对不同模型，稍微变化一点是有益的。
为了让模型最终的bitrate t更容易被预测和控制，发现采用clip rate term的方法是有效的。即，替换entropy term βH 为 max(t, βH)，即当 entropy term＜t时就会被关闭。我们发现这样并不会伤害性能。

为了获得不同码率的模型，可以通过调节 目标bitrate **t**和通道数 **K**来实现，当采用一个适度大的β=10时。我们采用了一个small regularization来稳定训练。

我们训练了6个epochs，那大约花费了24h(每个模型)在单个GPU上。

对于P，LR=1e-4，采用同样的衰减策略。

**Datasets**

ILSVRC12

预处理: random 160x160 crops, random flip them
从ImageNet中留出了100张图片作为testing set, **ImageNetTest**
进一步，在**Kodak**上test了模型。

为了获得high-quality full-resolution图片上的性能，也在数据集**B100**和**Urban100**上做了测试，这两个通常被用在super-resolution上。

**Other codecs**
比较了
JPEG: libjpeg
JPEG2000: Kakadu implementation
BPG: which is based on HEVC，先进的视频编码 Better Portable Graphics

**Comparison**

[12] O. Rippel and L. Bourdev. Real-time adaptive image compression.

**Results**


Kodak

![018eb29791783de8a1f8a7426351c763.png](https://a.photo/images/2018/04/14/018eb29791783de8a1f8a7426351c763.png)

注意到，R&B和Johnston也是直接maximize (MS-)SSIM,而其它方法都是minimize MSE.


**Ablation Study**

Context model:

验证context model的效果，做了如下实验:

训练了一个没有entropy loss的auto-encoder，即β=0, 使用了L=6个centers和K=16个channels.
在Kodak数据集上，该模型的avg ms-ssim是0.982, avg bpp是0.646(每个symbol假设需要log2(L)=2.59bits),然后我们固定该auto-encoder，然后训练了3个不同的context model:
1. 一个zeroth order的context model,其使用了直方图来估计L中每个符号的概率;
2. 一个first order(one-step prediction) context model,其使用了一个条件直方图来估计符号的概率(给定前一个符号的条件下);
3. P

可以发现,P可以减少rate 10%左右.

[![05a86c4544dcf32421a51d12f169e23f.png](https://a.photo/images/2018/04/14/05a86c4544dcf32421a51d12f169e23f.png)](https://a.photo/image/hQDU)


Importance map


## 5. Discussion

## 6. Conclusion


In this paper, we proposed the first method for learning
a lossy image compression auto-encoder concurrently with
a lightweight context model by incorporating it into an entropy
loss for the optimization of the auto-encoder, leading
to performance competitive with the current state-of-the-art
in deep image compression [12].

未来工作可能探索更heavy-更powerful的context models,它可以进一步提升压缩性能，并且允许对自然图像以"有损"方式来抽样：即先根据context model 来sample一个z\^,再解码。


# Supplementary Materials

## 3D Probability Classifier

masked 3D convolutions to enforce the causality constraint.

在2D CNN中，标准的2D卷积用在filter banks




![QQ20180414212244.png](https://a.photo/images/2018/04/14/QQ20180414212244.png)

如左图所示, 一个WxHxC_in维度的tensor被映射到一个W'xH'xC_out维度的tensor，使用C_out个banks of C_in 2D filters, 也就是filters可以被表示为fw x fh x C_in x C_out维度的tensors,注意到C_in个channels全部都被使用,这就违反了causality:当我们encode时,我们是逐个逐个通道来处理的.

使用3D卷积, 另一个维度--深度D被引入.一个WxHxDxC_in维度的tensor被映射到一个W'xH'xD'xC_out维度的tensor, filters可以表示为fw x fh x fd x C_in x C_out维度的tensor.(可以看出,深度也是滑动的,depth和width和height没有什么区别,都是data本身的维度).
对于P,我们采用的3D-CNN,输入是W x H x K维度的feature map z\^, 对于第一层,深度D采用全部深度K,C_in=1(因为只有1个3D image channel).


![QQ20180414213514.png](https://a.photo/images/2018/04/14/QQ20180414213514.png)

How we mask the filters in P?

简化而使用c x c context around z\^.
对于第一层和后续层的mask有所细微差别,第一层不能直接利用待编码符号,后续层可以利用该位置(因为它包含了最近的一圈的信息),即第一层是<,后续层是<=.

![QQ20180414215002.png](https://a.photo/images/2018/04/14/QQ20180414215002.png)

We can use P to
encode z\ˆ by iterating over z\ˆ in such blocks, exhausting first
axis w, then axis h, and finally axis d (like in Algorithm 1).


## Visual Examples

BPG倾向于高频信息做的更好(due to its more local approach),我们的方法有点Over-Blurring. 另一方面, BPG倾向于丢弃low-contrast high frequencies而我们的方法倾向于保留它.这可能是因为BPG是用MSE优化,而我们的方法使用MS-SSIM优化.

