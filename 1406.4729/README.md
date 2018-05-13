download: [https://arxiv.org/pdf/1406.4729]

title: 
Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[TOC]

---

### Abstract

移除CNN中对输入的size为固定size的约束, 产生一个fixed-length的representation. 通过空间金字塔池化操作(SPP). 在最后一个conv layer上添加了一个SPP layer.该SPP layer可以产生固定长度的输出，然后可以送入后续的fc层。

SPP(或者是spatial pyramid matching or SPM)是BoW(Bag-of-Words)模型的扩展: 将图片划分为从finer to coarser levels and aggregate local features in them.


### 1. Introduction

![CD8HIS.png](https://s1.ax1x.com/2018/05/13/CD8HIS.png)

### 2. Deep networks with spatial pyramid pooling

#### 2.1 Convolutional layers and feature maps

...

#### 2.2 The spatial pyramid pooling layer

spatial bin size 正比于 Image size
所以spatial bin number is fixed.

输出的维度是 kM， 其中k是filters的数量, M是number of bins.

![CDGNeP.png](https://s1.ax1x.com/2018/05/13/CDGNeP.png)

global pooling --- the coarsest pyramid level has a single bin that covers the entire image.

#### 2.3 training the network

**Single-size training**

feature maps : axa
pyramid level: nxn bins
win = [a/n]↑
str = [a/n]↓

[]↑ is ceiling
[]↓ is floor

l-level pyramid

**Multi-size training**

two networks share parameters

### 3. SPP-Net for Image Classification

#### 3.1 Experiments on ImageNet 2012 Classifictaion

#### 3.2 Experiments on