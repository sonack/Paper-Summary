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

![_2018-04-17_11-00-52.png](https://a.photo/images/2018/04/17/_2018-04-17_11-00-52.png)


quantization bin size is always one
and the representing values are the centers of bins
![_2018-04-17_11-01-44.png](https://a.photo/images/2018/04/17/_2018-04-17_11-01-44.png)

![_2018-04-17_11-04-50.png](https://a.photo/images/2018/04/17/_2018-04-17_11-04-50.png)

marginal density: 边缘密度


Replace the quantizer with an additive i.i.d uniform noise source △y, 它的width和quantization bins的宽度一样(1)。y~ = y + △y 的概率分布函数就是概率质量函数(质量和密度的区别在于离散/连续随机变量)q的一个连续的松弛形式. 在整数点，两者一样.

![_2018-04-17_11-08-29.png](https://a.photo/images/2018/04/17/_2018-04-17_11-08-29.png)


box-car filter?

![_2018-04-17_11-20-32.png](https://a.photo/images/2018/04/17/_2018-04-17_11-20-32.png)

其中φ是analysis的参数, Θ是synthesis的参数, 向量Ψ_i参数化了这个piecewise linear approximation of Pyi~, trained jointly with Θ and φ.

### 3.1 Relationship to variational generative image models

隐含变量 y
观测变量 x
我们有若干观测 P(x|y)， 希望寻找后验 P(y|x), 可以用 Bayesian variational inference 方法， 用另一个概率分布 q(y|x) 来逼近 P(y|x), 通过最小化KL距离:

![_2018-04-17_11-32-09.png](https://a.photo/images/2018/04/17/_2018-04-17_11-32-09.png)

(const 是关于x的概率， 因为x是确定观测的， 所以概率是常数)

这等价于我们松弛的rate-distortion optimization problem, distortion measure as MSE, 如果我们定义generative model如下:

![_2018-04-17_11-35-36.png](https://a.photo/images/2018/04/17/_2018-04-17_11-35-36.png)

Θ就是synthesis的参数,λ是trade-off.
给定y~， x是由y~决定的分布产生的.

y~是由Ψ控制的.

近似的后验如下:

![_2018-04-17_11-41-20.png](https://a.photo/images/2018/04/17/_2018-04-17_11-41-20.png)

U(yi~; yi, 1) 就是中心在yi的,长度为1的均匀概率分布.

这样的话，KL divergence中的第一项就是constant(因为是均匀分布), 第二项就意味着distortion, 第三项意味着rate.


## 4. Experimental Results

perturbation


## 5. Discussion

...



