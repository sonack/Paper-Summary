**Download:**   [https://arxiv.org/pdf/1804.06919]

**Title:**  Video Compression through Image Interpolation

**Authors:**    Chao-Yuan Wu, Nayan Singhal, Philipp Krähenbühl

---

[TOC]

---

## 0. Abstract

Video compression is repeated image interpolation.

outperforms H.261, MPEG-4 Part 2, and performs on par with H.264

## 1. Introduction

block motion estimation, residual color patterns, encoding using discrete cosine transform and entropy coding ...

the overall is not jointly optimized

end to end deep learning

This paper presents a first end-to-end trained deep videos codec.

We frame video compression as repeated image interpolation, and draw on recent advances in deep image generation and interpolation.

首先编码一系列的锚帧(关键帧), 用标准的deep image compression.

Our codec then reconstructs all remaining frames by interpolating between neighboring anchor frames.

然而，这种关键帧之间的插值不是唯一的，因此我们额外提供小的，可压缩的code到interpolation network来消除不同的插值方式的可能.

The main challenge is the design of a compressible image interpolation network.

vanilla U-net 

datasets:
uncompressed video

VTL(Video Trace Library)
Ultra Video Group(UVG)
a subset of the Kinetics dataset


## 2. Related Work

最简单的codec, 比如motion JPEG或者GIF, 独立地编码每一帧, 因此严重依赖图像压缩算法。

### Image Compression

对于视频，关键的一个冗余: 时间。

### Video Compression

Hand-designed video compression algorithms: such as H.263, H.264, HEVC(H.265) 建立在2个简单的ideas上:

1. 把每一帧分解为pixels的块, 即macroblocks, 把帧分为image frames(I) 和 referencing image(P or B). I-frames直接使用image compression算法来压缩图像, 视频编码中的大多数字节节省来自于referencing frames. P-frames从前面的帧s中借颜色值, 它们保存了一个motion estimate和一个高度可压缩的difference image为每一个macroblock. B-frames额外还允许Bidirectional referencing.只要没有循环引用。

H.264和HEVC都是以一种层次方式来编码视频. I-frames组成了层次的top, 在连续的level, P-和B-frames都在更高级别的levels上reference decoded frames.

### Image interpolation and extrapolation

图像的interpolation是在两个参考帧之间，寻找一个未曾见过的帧的近似。
大部分的Image interpolation networks建立在一个encoder-decoder网络架构之上，来移动pixels through time.

图像的extrapolation更加有野心(ambitious), 它从一些帧(或者静止帧)来预测a future video.

这些方法在small timesteps上都工作的挺好，但是在larger timesteps上就很难做好了, 此时interpolation和extrapolation都不再唯一, 额外的side information is required。本工作，扩展了image interpolation并将其与少量的可压缩的side information的bits相结合, 来重建原始video.


## 3. Preliminary

I(t) ∈ R(WxHx3) 代表 一系列帧 t ∈ {0, 1, ...}.

目标是压缩每一帧I(t)为一个二进制编码b(t)∈{0,1}\^N_t.

一个encoder E:  {I(0), I(1), ...} --> {b(0), b(1), ... }

一个decoder D: {b(0), b(1), ...} --> {I'(0), I'(1), ... }


**Image Compression**

Toderici: encodes and reconstructs an image progressively over K iterations.

**Video Compression**

![C1J0df.png](https://s1.ax1x.com/2018/04/25/C1J0df.png)

## 4. Video Compression through Interpolation

[![C1JRLq.md.png](https://s1.ax1x.com/2018/04/25/C1JRLq.md.png)](https://imgchr.com/i/C1JRLq)


选择每n帧就作为一个I-frame.

n=12 

### 4.1 Interpolation network

最简单的形式, 所有的R-frames都用一个blind interpolation network来在两个关键帧I1和I2之间插值.

context network C:

I --> {f(1), f(2), ... }

