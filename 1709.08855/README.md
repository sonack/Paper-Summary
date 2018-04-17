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
