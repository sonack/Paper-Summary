download:   [https://arxiv.org/pdf/1611.01704.pdf]

title:  End-to-end Optimized Image Compression

authors:   Johannes Ballé, Valero Laparra, Eero P. Simoncelli

[TOC]

## 0. Abstract

描述了一种image compression method, 包括

* a nonlinear analysis transformation
* a uniform quantizer
* a nonlinear synthesis transformation

local gain control

relaxed loss may be interpreted as the log likelihood of a generative model

## 1. Introduction

transform coding:

linearly transforming the data ==>  suitable continuous-valued representation ==> quantizing its elements independently ==> encoding the resulting discrete representation using a lossless entropy code.

eg:

JPEG:   discrete cosine transform on blocks of pixels
JPEG2000:  multi-scale orthogonal wavelet decomposition.

Generalized divisive normalization (GDN)

## 2. Choice of Forward, Inverse, and Perceptual Transforms


![CejCnI.png](https://s1.ax1x.com/2018/04/16/CejCnI.png)


analysis transform g_a 包含 3个stages of convolution, subsampling and divisive normalization.


![CeOoeP.png](https://s1.ax1x.com/2018/04/16/CeOoeP.png)
表示 第ith input channel, kth stage, spatial location (m,n).

输入image vector x 对应 ![_2018-04-16_17-38-34.png](https://a.photo/images/2018/04/16/_2018-04-16_17-38-34.png)
输出vector y 是 ![CeOxLq.png](https://s1.ax1x.com/2018/04/16/CeOxLq.png)

![CeX9oT.png](https://s1.ax1x.com/2018/04/16/CeX9oT.png)

![CeXEl9.png](https://s1.ax1x.com/2018/04/16/CeXEl9.png)、

![_2018-04-16_17-34-28.png](https://a.photo/images/2018/04/16/_2018-04-16_17-34-28.png)


类似的, synthesis transformation g_s consist of 3 stages.

与analysis transformation中的每个stage中的操作相反，downsampling 换成 upsampling, GDN 替换为 一个近似的逆操作 IGDN

![CeXcmq.png](https://s1.ax1x.com/2018/04/16/CeXcmq.png)

之前的工作，使用了一个perceptual transform g_p,  这里使用identity和MSE， 便于比较，且可用于color images.


## 3. Optimization of non-linear transform coding model

