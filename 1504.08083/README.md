download: [https://arxiv.org/pdf/1504.08083]

title: 
Fast R-CNN

authors: Ross Girshick

[TOC]

---

### Abstract

https://github.com/rbgirshick/fast-rcnn

### 1. Introduction

### 1.1 R-CNN and SPPnet

R-CNN的缺点:

1. 训练是一个多阶段的流程: 首先finetune一个ConvNet, 然后还要训练SVM. 第三个训练阶段是bounding-box regressors的训练。

2. 训练需要耗费巨大的时空资源: 为了训练SVM和bounding-box regressor, 特征(每张图片的每一个object proposal)都需要被抽取然后写入磁盘。

3. 目标检测很慢: VGG16 takes 47s/image (on a GPU).

SPPnet cannot update the convolutional layers that precede the spatial pyramid pooling.

### 1.2 Contributions

Fast R-CNN advantages:

1. 更高的检测质量 (mAP) 比起 R-CNN, SPPnet
2. 训练是单阶段的, 使用一个multi-task loss
3. 训练可以同时更新所有网络层
4. 不需要缓存feature, 节省了磁盘空间

### 2. Fast R-CNN architecture and training

![CrQQVU.png](https://s1.ax1x.com/2018/05/14/CrQQVU.png)

input: an entire image and a set of object proposals

首先经过几个conv层和max pooling层产生conv feature map, 然后对于每一个object proposal 一个 region of interest(RoI) pooling layer从feature map上抽取一个fixed-length feature vector.

每个feature vec被送入一系列fc层, 最终产生2路分支: 一个产生分类(K object + background)的softmax probability estimates; 另一路对K个classes中的每一个产生4个实数值, 每4个值都编码了对K类中的某一类的bounding-box的位置进行修正的信息.

#### 2.1 The RoI pooling layer

RoI max pooling

HxW are layer hyper-parameters that are independent of any particular RoI.


#### 2.2 Initializing from pre-trained networks

#### 2.3 Fine-tuning for detection

**Multi-task loss**

![Crlisx.png](https://s1.ax1x.com/2018/05/14/Crlisx.png)

L_cls(p,u) = - log p_u 是 真实class u 的 log loss.

L_loc  

u=0: background, L_loc is ignored

![CrlYFg.png](https://s1.ax1x.com/2018/05/14/CrlYFg.png)

![Crl0O0.png](https://s1.ax1x.com/2018/05/14/Crl0O0.png)

robust L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPPnet.

Normalize the GT regression targets v_i to have 0 mean and 1 variance.

λ=1.

**Mini-batch sampling**

**SGD hyper-parameters**

#### 2.4 Scale invariance


brute force learning (single-scale)
image pyramids (multi-scale)

### 3. Fast R-CNN detection

#### 3.1 Truncated SVD for faster detection

...



