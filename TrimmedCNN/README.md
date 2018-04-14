title: Enlarging Context With Low Cost: Efficient Arithmetic Coding with Trimmed Convolution

info: Anonymous ECCV submission, Paper ID 340


[TOC]

## 0. Abstract

算术编码的一个关键问题就是给定context预测当前码元(即给定之前已经编码的符号后),通常通过构建一个look-up table(LUT)来解决该问题.
但是当context的长度增加时,LUT的size是随着指数型增加的,因此该方法在处理大的context时候会受限.
提出了trimmed convolutional network for arithmetic encoding(TCAE),trimmed目的是尊重编码顺序和context之间的依赖顺序,效率较高,所有码位符号的概率预测都可以在一趟forward pass内完成(通过一个全卷积网络).

为了加速解码过程,一个slope TCAE模型被提出,来将codes从3D code map拆分为几个blocks,并且消除了同一个block内的codes之间的依赖关系,从而可以充分利用并行计算,它加速了60x解码过程.

实验表明,TCAE和slope TCAE在loseless grayscale image compression取得了更好的压缩率,并可以被用在CNN-based image compression中,实现了SOTA的rate-distortion performance, 带有实时编码速度.


## 1. Introduction

熵编码的关键问题: 预测当前码元

在JPEG中, symbols的频率直接采用count获得,然后用经典哈弗曼编码.

在JPEG2000中, EBCOT coder被用来model the context来估计概率,然后Binary Arithmetic Coding-MQ-Coder被用来压缩codes.

在H.264/AVC, CABAC(context-adaptive binary arithmetic coding)被引入,用来利用前面2个编码完的符号来预测概率,再进行arithmetic coding. 它采用了LUT方法.


PixelCNN和PixelRNN模型验证了DNN用来捕获像素之间高度复杂和长程的依赖的能力.

TCAE引入了一种新的trimmed conv operation来加速概率预测.它利用binary mask来避免使用某些输入值.

slope TCAE, 
3D code maps --> divided into several blocks(block间还要串行,但是block内可以并行). 3D code map x 的坐标为(i,j,k), 则第t个code block 被定义为

CB_t(x) = {x[i,j,k] | i+j+k=t}.


JPEG-LS (LossLess)
JPEG2000-LS

之前的CWCNN加上TCAE后,encode可以做到real time, decode可以比之前快60x.


contribution summary:

1. Trimmed Conv used incorporated with fully convolutional network can perform prediction to all bits within one single forward pass.
2. TCAE encoder用来encoder grayscale images或者CNN-based lossy compression system的intermediate codes.
3. Slope TCAE来打破一个块内的依赖,可以并行decode,它加速了decoding process 60x.
4. 实验表明对于无损灰度图压缩,TCAE比PNG和JPEG2000-LS能达到更高的压缩率, 对于CNN-based lossy compression system, TCAE可以在不牺牲rate-distortion performance的前提下达到real time encoding speed.


## 2. Related Work

### 2.1 Lossless image compression standards


JPEG-LS take use of the LOCO-I algo, 它采用了median edge detection predictor来预测the image,然后压缩the residual with context-based entropy coding.

JPEG2000-LS, based on reversible integer wavelet transform(biorthogonal 3/5).

PNG 首先利用一个无损filter来转换图像, and then compress the transformed data with DEFLATE algo, DEFLATE是一种LZ77和Huffman coding的combination.

TIFF和GIF都用了LZW(Lemple-Ziv-Welch)算法,它是一种dictionary based method for lossless image compression.

### 2.2 Entropy encoding
Entropy encoding is a lossless data compression scheme which has played an essential role in both lossy and lossless image compression systems. 

* Run length encoding: consecutive symbol sequence are stored as the symbol and its count.
* Huffman coding: Variable-length coding scheme.
* Golomb-Rice coding: input ==> quotient + remainder ==> truncated binary encoding to encode the remainder.
* Arithmetic coding: predict probability of the current symbol, divide the current interval into sub-intervals for encoding the updated sequence.

### 2.3 Deep networks based entropy encoding

[TODO]












