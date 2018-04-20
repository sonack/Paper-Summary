download:   [https://arxiv.org/pdf/1703.00395.pdf]

title:  Lossy Image Compression with Compressive Autoencoders

authors: Lucas Theis, Wenzhe Shi, Andrew Cunningham, Ferenc Huszár

---

encoder (powerful)
decoder (less complex)

quantization --> not differentible

rate-distortion --> rounding-based

performance: similar to or better than JP2

CAE:

encoder f
decoder g
probabilistic model Q

![_2018-04-20_14-17-34.png](https://a.photo/images/2018/04/20/_2018-04-20_14-17-34.png)


entropy rate estimation

![_2018-04-20_14-20-34.png](https://a.photo/images/2018/04/20/_2018-04-20_14-20-34.png)

![_2018-04-20_14-21-13.png](https://a.photo/images/2018/04/20/_2018-04-20_14-21-13.png) (Jensen's Inequality)

An unbiased estimate of the upper bound is obtained by sampling **u** from the unit cube [-.5, .5]\^M. 

model the q, independent Gaussian scale mixtures(GSMs)

![_2018-04-20_14-28-13.png](https://a.photo/images/2018/04/20/_2018-04-20_14-28-13.png)

i,j 遍历空间位置
k 遍历channels

**Variable Bit Rates**

![_2018-04-20_14-36-02.png](https://a.photo/images/2018/04/20/_2018-04-20_14-36-02.png)


Incremental Training

[![_2018-04-20_14-40-44.png](https://a.photo/images/2018/04/20/_2018-04-20_14-40-44.png)](https://a.photo/image/jymo)

additional binary mask



![_2018-04-20_14-42-16.png](https://a.photo/images/2018/04/20/_2018-04-20_14-42-16.png)


downsampling: using conv
upsampling: performed using sub-pixel convolutions

only 2 residual blocks are shown

CxKxK 代表 KxK conv with C filters

/2 表示 conv 的 stride 或者 upsampling factor (sub-pixel conv)

黑色箭头表示conv followed by leaky rectifications, 透明箭头表示没有non-linearity.
