**Download:**   [https://arxiv.org/pdf/1804.04593]

**Title:**  Deformation Aware Image Compression

**Authors:**    Tamar Rott Shaham, Tomer Michaeli

---

[TOC]

---

## 0. Abstract

现存的有损压缩算法，浪费了大量bit在描述图像中每个细节的几何信息，然而，人眼对微小的局部的平移是不敏感的。 本文提出了一种 变形-不敏感 错误衡量指标，它可以与现存的任何压缩算法相结合，即: 最优压缩牵扯到对输入图像进行微小的变形，以便使其更容易被压缩，经过人眼不可察觉的微小形变后，可以使codec保留之前会丢弃掉的细节，本文把codec作为一个"黑箱"使用，这些可以与其集成的压缩系统包括JPEG, JPEG2000, WebP(2010), BPG(2014), 和最近的基于深度网络的方法。

## 1. Introduction

![_2018-04-21_12-56-51.png](https://a.photo/images/2018/04/21/_2018-04-21_12-56-51.png)

PDE-based approaches

现在的距离衡量指标(包括感知指标，如SSIM)的主要局限性在于它们都对两张图像的微小的形状和对象的不对齐太过于敏感。

因此，在这些指标下，如果指标要更好，就要花费大量的比特来精确编码每个细节的几何信息，这是浪费，因为人类视觉系统对于微小的几何形变是不敏感的，只要整体的语义被保留。

本文提出了一个新的error measure, 它对微小的平滑的形变是不敏感的，它有两个主要优势:

1. 很容易集成到其他任何压缩方法中。
2. 更符合人类的视觉感知(通过用户调查得到)，保留细节方面有很大的提高。


high compression ratios ==> low bitrate

## 2. Related Work

comply: 遵从

using deformations for idealizing images

The idea of measuring similarity up to deformation has been proposed in the context of image recognition and face recognition, but this approach has never been exploited for image compression.


## 3. Deformation Aware Compression

SSD(mse): sum of squared differences


DASSD(Deformation aware SSD):

考虑两个图像x和y， 如果存在一个smooth deformation T 使得 T(y) 和 x相似，则认为x和y是相似的。

![_2018-04-21_13-40-18.png](https://a.photo/images/2018/04/21/_2018-04-21_13-40-18.png)

这里的ψ用来惩罚不平滑的形变。
注意，计算DASSD需要解一个optical-flow problem来决定最好的warp y onto x.

为了允许复杂的deformation, 我们采用一个 nonparametric flow field (u,v), 即

![_2018-04-21_13-46-28.png](https://a.photo/images/2018/04/21/_2018-04-21_13-46-28.png)

define the penalty ψ(T) to be a weighted Horn and Schunk regularizer

![_2018-04-21_14-16-32.png](https://a.photo/images/2018/04/21/_2018-04-21_14-16-32.png)

## 4. Algorithm

