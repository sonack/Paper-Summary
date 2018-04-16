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

PixelCNN和PixelRNN在图像生成和建模复杂长程依赖上展示了巨大威力，但它们都只适用于2D图像，而不能用于3D code maps. Toderici et al. 学习一个binary recurrent network来建模来自相同coding plane和之前coding planes的codes的context. 然而，每趟forward pass都只能估计一个bit的概率，这使得binary RNN estimator计算非常没有效率。之前李穆的工作引入了一个convolutional entropy estimator来建模3D code maps, 通过抽取一个特殊定义的context cube, 然后用一个小的卷积网络来处理它，相比于binary RNN, convolutional estimator是相对更有效率的， 但是由于context的重叠，重复计算仍然是不可避免的。


## 3. Trimmed Convolutional Arithmetic Encoding


![_2018-04-15_13-57-35.png](https://a.photo/images/2018/04/15/_2018-04-15_13-57-35.png)

### 3.1 Arithmetic coding

给定一个符号串，熵编码为经常出现的符号分配更少的比特，为不经常出现的符号分配更多的比特。 不同于其他的熵编码算法, 如 Huffman coding, 算术编码把整个符号串编码成一个[0,1]区间的单个实数。 如果一个coding system中有k个符号, 给定一个新的符号x_i, 和当前区间, 算术编码首先预测概率分布 P(x_i=t|x_(i-1), ... , x0)， 然后当前区间根据预测的概率被继续划分为若干个子区间。 为了继续编码后续符号序列，当前区间根据新符号的groundtruth和之前预测的概率被更新，如下图所示:

编码(0,2,3)， 符号为(0,1,2,3), 离散的概率分布为(0.6, 0.2, 0.1, 0.1)


![_2018-04-15_14-13-56.png](https://a.photo/images/2018/04/15/_2018-04-15_14-13-56.png)

### 3.2 Coding schedule and context of 3D cuboid

3D binary cuboid:

X = { x[i,j,k] | 0 <= i <= W-1, 0 <= j <= H-1, 0 <= k <= C-1 }

![_2018-04-15_14-19-23.png](https://a.photo/images/2018/04/15/_2018-04-15_14-19-23.png)

编码顺序:

(1). x[i+1,j,k] 在 x[i,j,k] 之后编码， 直到i=W-1
(2). 当i=W-1时， x[0,j+1,k] 在之后编码， 直到j=H-1
(3). 当i=W-1,j=H-1时，x[0,0,k+1] 在之后编码


算术编码的关键问题是从context预测一个符号的概率。 给定一个3D cuboid中的位置(p,q,r), 它的full context 记为 CTX(x[p,q,r]), 考虑到X的编码顺序, CTX(x[p,q,r])可以定义为如下:

CTX(x[p,q,r]) = {x[i,j,k] | {k < r} V {k = r, j < q} V {k = r, j = q, i < p}}.

关键在于CTX(x[p,q,r])的长度不是固定的，并且随着(p,q,r)在改变， 使它很难学习。

注意到越靠近待编码bit的context就扮演着越重要的角色， 因此我们采用一个固定长度的context定义，如下所示:

CTXf(x[p,q,r]) = {x[i,j,k] | r - c <= k < r , |i - p| <= w, |j - q| <= h} V {k = r, q - h <= j < q, |i - p| <= w} V {k = r, j = q, p - w <= i < p}.



大的context(即大的w,h,c)会对概率预测有帮助，

### 3.3 Trimmed convolutional networks

context cuboid:

(2w+1)x(2h+1)x(c+1)

幸运的是， 每个位置都有两种值，i.e. 原始值(x[p,q,r]) 和 默认值(for non-encoded bit)， 给定待编码体素 x[p,q,r] , non-encoded bits的位置都是固定的. 不失一般性，让默认值为0，我们可以引入一组trimmed conv operators来组成一个全卷积网络来context modeling.

![_2018-04-15_15-16-00.png](https://a.photo/images/2018/04/15/_2018-04-15_15-16-00.png)

**Notes:**
l = p + i
m = q + j
n = r + k

t = r (上下延伸的深度)
-t <= k <= C - t - 1   上下界相差固定(C-t-1-(-t) = C-1),  t 控制 上下平移

t = 0 ==> 0 <= k <= C - 1  
t = r ==> -r <= k <= C - r - 1 


对w0来说, non-encoded bits的位置是固定的,预先就定义好的.因此我们可以引入mask m of {0,1} 去将它们从convolution中排除掉. 如果x[p+i, q+j, r+k]是encoded在x[p,q,r]之前,则m[i,j,k]定义为1，否则为0.

m的设置对于input layer和hidden layer是不同的.

For input layer:

![_2018-04-16_14-38-55.png](https://a.photo/images/2018/04/16/_2018-04-16_14-38-55.png)

For hidden layer:
x\^d [p,q,r] (d >= 1) 我们认为其只包含x[p,q,r]的context的信息， 并不应该再被排除。
因此:

![CeJnUI.png](https://s1.ax1x.com/2018/04/16/CeJnUI.png)

(区别在于<和<=)

扩展到multiple convolution kernel form:

假设输入有 g_in 组 feature maps (d-layer)

![CeYVzT.png](https://s1.ax1x.com/2018/04/16/CeYVzT.png)

输出有 g_out 组

![CeYnL4.png](https://s1.ax1x.com/2018/04/16/CeYnL4.png)


![_2018-04-16_14-46-59.png](https://a.photo/images/2018/04/16/_2018-04-16_14-46-59.png)


m\^d 是 第d层 的mask(mask与位置pqr有关), w\^(d,g,g')是连接X\^(d,g)和X\^(d+1,g')的convolution kernel.

Notes: 这里的3D kernel其实就是一个3D的weight matrix, 而少了channel一维度 (类比于2D conv).

### 3.4 Slope TCAE

利用TCAE虽然编码时可以并行，但在解码时, 为了预测xi的概率, p(x_i|x_(i-1), ... , x_0), 需要依赖于x_(i-1), ... , x_0, 从而使得解码必须一个位置一个位置的串行进行。

为了加速decoding process, 我们可以打破codes之间的某些依赖关系，即，为了对于x_i, 它不依赖于所有的 x_(i-1), ... ,x_0, 而只是它们中的一部分。

第t个 code block 定义如下:

![_2018-04-16_15-00-49.png](https://a.photo/images/2018/04/16/_2018-04-16_15-00-49.png)

因此， x_i 如果 属于 CBt(x), 则x_i的概率可以表示为 ![_2018-04-16_15-02-30.png](https://a.photo/images/2018/04/16/_2018-04-16_15-02-30.png)

同一个块内的context是相同的, 并且可以同时被预测出来.
因为同一个块内的codes恰好落在一个cuboid的slope plane上，因此我们称这种并行context modeler为 slope TCAE.

**With slope TCAE, the context should be modified.**

The context for code x[p,q,r]


![CetTUA.png](https://s1.ax1x.com/2018/04/16/CetTUA.png)

对于hidden layer来说, < 变为 <=.

coding schedule 定义为:

从CB[0] 到 CB[H+W+C], 在CB[t]的内部, 我们首先按照k的升序, 然后是i.

mask 的 定义 也修改为:

![CeNF2V.png](https://s1.ax1x.com/2018/04/16/CeNF2V.png)


### 3.5 Model objective and learning

给定模型参数 W={w\^(d,g,g')}, trimmed conv network的输出写作F(X;W), 其包含m个parts, m是code maps中的编码符号数量, 我们采用算术编码后的codes的长度作为模型的目标函数:

![CeNNad.png](https://s1.ax1x.com/2018/04/16/CeNNad.png)


p,q,v 遍历 体素位置, s(x,t) = 1 when x = t, 0 otherwise.

根据香农的信息论，压缩率(compression ratio)定义为the ratio between uncompressed size and compressed size.

![CeNBxf.png](https://s1.ax1x.com/2018/04/16/CeNBxf.png)

因此，等价于直接优化压缩率。

训练细节:

ADAM solver
lr:  3x10^(-4), 1 x10^(-4), 3.33 x 10^(-5), 1.11 x 10^(-5)  (逐步除以3)

更小的学习率直到较大的学习率停止降低loss后就会被采用.

## 4. Experiments

3组实验

1. 第一组: grayscale image lossless compression  为了满足3D binary cuboid的需求， 我们采用8-bit表示作为输入.

2. 第二组: 我们用该grayscale image predictor做了Image Inpainting.

3. 第三组: 是与CNN-based lossy compression incorporation

### 4.1 Network architecture and parameter setting

TCAE:

![CeUCee.png](https://s1.ax1x.com/2018/04/16/CeUCee.png)

Trimmed Residual Block: consists of two trimmed convolution layers with each followed by the ReLU nonlinearity. and the skip connection is also added from the input to the output of the residual block.

input shaped:  chw x 1
output shaped: 2D matrix with the size of chw x m

训练数据:  10,000 张 高质量图片 来自 ImageNet
test on Kodak PhotoCD image dataset.

因为TCAE是全卷积网络, it can be trained and tested using images with any size.

### 4.2 Lossless grayscale image compression

Matlab2015b

![_2018-04-16_15-53-35.png](https://a.photo/images/2018/04/16/_2018-04-16_15-53-35.png)

### 4.3 Image Inpainting

我们从原始图像的右下角corrupt一个rect area(1/9 size)，然后填充它利用TCAE context predictor.


### 4.4 CNN-based lossy image compression


![_2018-04-16_16-01-04.png](https://a.photo/images/2018/04/16/_2018-04-16_16-01-04.png)

25 fps

### 4.5 Slope TCAE vs. TCAE


slope TCAE only has small drop both in grayscale image compression and compression 3D code maps


在decoding period, 原始的convolutional entropy encoder花了大约212.32秒去解压3D binary code maps with the size of 94 x 68 x 64.


Cropping the 3D binary codes ? 

## 5. Conclusion

...

