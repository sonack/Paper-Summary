download: [https://arxiv.org/pdf/1709.08855.pdf]

title: Learning to Inpaint for Image Compression

authors: Mohammad Haris Baig, Vladlen Koltun, Lorenzo Torresani

info: NIPS 2017


[TOC]

## 0. Abstract

(a). predicting the original image data from residuals in a multi-stage progressive architecture facilitates learning and leads to improved performance at approximating the original content;
(b). learning to inpaint (from neighboring pixels) before performing compression reduces the amount of information that must be stored to achieve a high-quality approximation. 

相同的质量，减少了60%的文件体积.

## 1. Introduction

progressive codes:  a sequence of representations that can be transmitted to improve the quality of an existing estimate (from a previously sent code) by adding missing detail.

本文章的主要贡献有2部分:

1. 传统的progressive encoders被优化来压缩residual errors in each stage of their architecture(residual-in, residual-out), 相反，我们的模型at each stage 从 之前stage的residual 来预测 原始图像 (residual-in, image-out). 我们发现这样使得优化更加容易，能产生更好的图像压缩。这种架构减少了用来恢复图像到相似质量时的需要存储的信息量18%，相比于传统的residual encoder.
2. 现有的深度架构并没有充分利用图像相邻patches之间的空间上的高度连续性。我们展示了如何设计和训练一个模型，它可以充分利用相邻区域的依赖性，通过从可获得的内容来做inpainting. 我们引入了多尺度的卷积,它可以sample context在不同尺度，以协助inpainting. 我们联合训练我们所提出的inpainting和compression模型，发现inpainting可以减少必须存储的信息量额外的42%.



## 2. Approach

progressive multi-stage encoder-decoder with S stages:

我们采用 convolutional-deconvolutional residual encoder (proposed by Toderici) 作为我们的reference model. 这个model抽取一个compact binary representation B from an image patch P. 这个binary representation，用来重建原始patch的近似, 包含了 从这个模型的S个stages中抽取的 一系列representations。即

B = [B1, B2, ... , Bs]

model的第一个stage抽取一个binary code B1 从输入patch P, 随后的每个stage都学习Bs， to model the compressions residuals R(s-1) 从之前的stage.

compression residuals R(s) 定义为:

![_2018-04-17_17-12-55.png](https://a.photo/images/2018/04/17/_2018-04-17_17-12-55.png)

其中Ms(R(s-1) | Θs) 表示在stage s中 获得的 reconstruction 在 modeling R(s-1) 时。

model 在 每一stage 都被分为一个encoder B 和 一个decoder D

目标函数:

[![_2018-04-17_23-18-07.png](https://a.photo/images/2018/04/17/_2018-04-17_23-18-07.png)](https://a.photo/image/jL5N)

R0(i) = P(i)
Rs(i) 表示 对于patch P(i) 的compression residual.


P  ==> P'
P-P' ==> P''
P-P'-P'' ==> P'''
...

P''' + P'' + P' ≈ P

不容易优化，因为gradients必须要经过很长的path,从后面的stage到影响前面的stage, 为了解决梯度消散的问题，我们研究了一类新的架构,叫做"Residual-to-Image(R2I)".

### 2.1 Residual-to-Image(R2I)

通过增加connections在后续stages之间，重新表述loss为预测原始图像,在每个stage的最后(类似于RNN)。这样，输入的是残差，输出的是图像。新的目标函数如下：

因为增加了connection,因此各个stage之间的输入是互通的，类似于RNN.
![_2018-04-17_23-29-43.png](https://a.photo/images/2018/04/17/_2018-04-17_23-29-43.png)

model的stage s输入compression residual R(s-1), 其计算公式是相对于原始图像，如下:

![_2018-04-17_23-33-19.png](https://a.photo/images/2018/04/17/_2018-04-17_23-33-19.png)

其中Ms-1是在stage s-1的对original data P的approximates.

为了满足完整图像的reconstruction在每个stage都能被produced(在只送入residuals的条件下),我们在相邻stages的层之间添加了connections. 这些连接允许后面的stages可以聚合前面层已经恢复的信息以恢复original image data.最终，这些连接使得模型更容易被优化。

除了帮助modeling原始图像，这些连接还扮演着两个重要角色:

1. 这些连接构成了residual blocks, 这鼓励了显式学习如何再生信息(其不能被前面那些stages产生).
2. 这些连接缩短了信息从后面的stages去影响前面的stages的路径，导致一个更好的联合优化。

问题在于,在哪里插入这样的connections,信息应该如何传播？

我们考虑2类connections:

1. predictive connections: 类似于resnet中的identity shortcuts. parameter free
每个stage的输出都是residual predictions of 当前stage和之前所有stages的 相加 在应用最终的non-linearity之前， 如下图b所示:

![_2018-04-18_09-56-39.png](https://a.photo/images/2018/04/18/_2018-04-18_09-56-39.png)



2. parametric connections

类似于resnet中的 projection shortcuts

卷积后再相加相邻之前stage的对应层
2 variants:  "full", "decoding"
如图c和图d


LSTM-based model of Toderici et al.代表了一个特殊的R2I network with full connections. 在section3我们会展示对于R2I models, decoding connections 比 full connections做的更好，并且提供一个直觉的解释。



### 2.2 Inpainting network

deep models for inpainting, are trained to predict the values of pixels in the region W\^ from a context region C\^.

相邻patch不是独立的

![_2018-04-18_10-15-23.png](https://a.photo/images/2018/04/18/_2018-04-18_10-15-23.png)

partial-context inpainting: 由于decoding的顺序，只有左和上的patches被解码。

我们训练inpainting的目标函数是:

![_2018-04-18_10-19-32.png](https://a.photo/images/2018/04/18/_2018-04-18_10-19-32.png)

#### 2.2.1 Architecture of the Partial-Context Inpainting Network

feed-forward architecture


to imporve ability of model:  use a multi-scale convolutional layer

dilated convolution to allow for sampling at various scales

对于每一个dialation factor, 每个multi-scale conv layer 都由 k 个 filters组成.
改变这个dilation factor 给了我们在不同尺度上分析内容的能力。

multi-scale convolutional layer 如上图b所示.

[如何理解空洞卷积?](https://www.zhihu.com/question/54149221)
增大感受野


多尺度的conv layer给了我们自由去在full-resolution(no striding or pooling)去传播内容，只需要很少的multi-scale layers就能够满足恢复整个区域的要求. 这允许我们可以训练一个相对较浅但是更有表达力的架构，它可以传递细粒度的信息(可能之前因为sub-sampling而丢失的那些信息)。这样的轻量高效的设计，对多stage compression model的联合训练是十分需要的。





#### 2.2.2 Connecting the Inpainting Network with the R2I Compression model

How to use the prediction of inpainting for assisting with compression



![_2018-04-18_10-48-13.png](https://a.photo/images/2018/04/18/_2018-04-18_10-48-13.png)

第一项L_inp, 对应原始的inpainting loss

R0(i) 表示 对于example i 的inpainting residual

![_2018-04-18_10-53-38.png](https://a.photo/images/2018/04/18/_2018-04-18_10-53-38.png)

注意到每个stage of the inpainting-based progressive encoder 直接影响了inpainting network学习到什么，我们把这个joint objective训练的model 叫做 "Inpainting for Residual-to-Image Compression" (IR2I).

使用stage2的近似来作inpainting, 并且在传输时，前两个stage作为一个更大的整体单元一起传输。

### 2.3 Implementation Details


ImageNet 6507 Image (as Balle)
Caffe
residual encoder and R2I models were trained for 60,000 iterations
joint inpainting network was trained for 110,000 iterations


Adam + MSRA init

lr = 0.001 and dropped after 30K, 65K, 90K of factor 10

All of our model were trained to reproduce the content of 32x32 image patches

each of our model has 8 stages, and every stage contributing 0.125 bpp (16 bytes each stage?)

Raiko:

![_2018-04-18_11-11-02.png](https://a.photo/images/2018/04/18/_2018-04-18_11-11-02.png)


inpainting network has 8 multi-scale conv layer and 1 standard conv layer for final prediction

每一个multi-scale conv 都包含24个filters for dilation factors 1,2,4,8  (ie. 6 filters per factor)

输入context region C of size 64 x 64, where the bottom right 32x32 is zeroed out and represents the region to be inpainted.

## 3. Results


Kodak 24张512*768图片

quality measured by ms-ssim(更高的值意味着更好的质量)

Bjontegaard-Delta metric 来计算平均bit-rate在所有的quality settings.

### 3.1 R2I - Design and Performance

the model with decoding connections does better than full connections, because:

在解码时候，只需要当前残差的信息即可，如果在encoder上也有connections,则也包含了已经被前面的stage捕获过的二进制表示信息，可能会加大单独的stage识别相关信息来重建的难度，因此导致优化更为困难。



R2I model improve at higher bit rates but not so much at lower bit rates;

lower bit rates:  depend on inpainting

### 3.2 Impact of Inpainting

joint training

### 3.3 Comparison with Existing Approaches

## 4. Conclusion and Future Work

1. incorporate entropy encoding
2. extend to video data


## 5. Acknowledgement
...

## Appendix

### A. Progressive vs Non-Progressive Coding Approaches

希望未来能研究针对progressive model的entropy estimation方法，将其加到训练目标中。

### B. Qualitative Analysis

定性分析
quantitative 定量

### C. Architecture Specification

所有的卷积层都使用 size 3x3 除了 Conv*, 它使用1x1.
所有的deconv都使用 size 2x2 的filters





