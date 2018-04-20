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

量化: 量化到bit精度B
比特平面的分解:  量化后的tensor y\^ 被转换为bianry tensor以便适合用lossless bitplane decomposition去编码:

![_2018-04-20_12-09-40.png](https://a.photo/images/2018/04/20/_2018-04-20_12-09-40.png)

动态算术编码: AAC
adaptive codelength regularization: ACR调整量化后的y\^分布以便达到一个目标期望的bit数

![_2018-04-20_12-12-22.png](https://a.photo/images/2018/04/20/_2018-04-20_12-12-22.png)

#### 3.2.1 Quantization


给定期望的bit精度 B, 我们量化我们的特征张量y到2^B个equal-size bins 为:

![_2018-04-20_12-14-14.png](https://a.photo/images/2018/04/20/_2018-04-20_12-14-14.png)

特例:

B=1 : ie binary quantization  ie. round
B=2 : [2*x]/2 0~0.5, 0.5~1   more precise, more smoother

We quantize with B = 6
...

#### 3.2.2 Bitplane Decomposition
...
#### 3.2.3 Adaptive Arithmetic Coding
...
#### 3.2.4 Adaptive Codelength Regularization

ACR
We regular our quantized tensor y\^
![_2018-04-20_13-00-54.png](https://a.photo/images/2018/04/20/_2018-04-20_13-00-54.png)

for iter t,  S是坐标差集合: {(0,1), (1,0), (1,1), (-1,1)} (左,上,左上,右上 四个邻居)

第一项惩罚tensor每个元素的大小，第二项惩罚与相邻邻居的偏差大小。

这使得AAC可以更好预测。

在训练时，我们不断调节标量参数αt的值来达到我们目标的codelength, 我们通过一个反馈回路来做到这点。我们用AAC来查看有效bits的平均数量，如果太大，就增加αt； 否则就减少αt. 在实践中，在几百个iterations中它就达到了平衡点，此后就可以一直维持它。

因此，我们有了一个调节的把手: BCHW in b, 目标有效bits数量 l(target).
可以通过增加channels或者spatial size of b ， 同时增加稀疏性。 我们发现 total-to-target ratio of BCHW / l(target) = 4工作的很好

### 4. Realistic Reconstructions via Multiscale Adversarial Training

#### 4.1 Discriminator design

take the generator as the encoder-decoder pipeline

将分类target和重建reconstruction 联合起来: 比较两者，通过询问两张图像中的哪一个是real one.

首先交换target和reconstrution在每个input pair到discriminator利用Uniform Probability

相反，不是在pipeline的最后产生一个output, 而是多个阶段,多个scale average.


#### 4.2 Adversarial training

动态调整，训练D还是reconstruction

L=0.8, U=0.95

accuracy α是一个running average over mini-batches with a momentum of 0.8.


### 5. Results
#### 5.1 Experimental setup

**Similarity metric**
ms-ssim
**Color space**

人类视觉系统对于亮度的变化比颜色更敏感，因此大多数的codec都把颜色表示为 YCbCr空间以将更多的带宽用来压缩亮度路明而不是色度。

我们既在RGB空间(各颜色通道相同权重), 也有在YCbCr空间(权重为Y:Cb:Cr = 6:1:1)

**Reported performance metrics**

compression performance + runtime

**Training and deployment procedure**

128 x 128 patches sampled at random from the Yahoo Flickr Creative Commons 100 Million dataset.

Adam  

3x10^-4  在训练过程中，以因子5减少2次
我们选择一个batch size=16, 每个模型训练400,000迭代。

#### 5.2 Performance
 
Average MS-SSIM;  固定BPP
Average Compressed file size;  固定MS-SSIM
Encode and Decode Timings;  固定MS-SSIM
相同BPP，不同方法的视觉重建效果的对比

**Test sets**

Kodak: 24  容易收到过拟合的影响，并且不能完全捕获自然图像的统计性质
<font color="red" >RAISE-1k dataset </font>: 包含1000张图片

resize each image to size 512 * 768 (如果是垂直的，则反过来)
要使用raw images,而不能是compressed images, 因为一个被某个codec压缩过的图像，再被压缩的话，展示的RD曲线会过好。

**Codecs**
**Performance evaluation**

### Supplementary Materials

The architecture of the discriminator

![_2018-04-20_13-55-54.png](https://a.photo/images/2018/04/20/_2018-04-20_13-55-54.png)

