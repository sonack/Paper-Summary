**Download:**   [https://arxiv.org/pdf/1608.05148.pdf]

**Title:**  Full Resolution Image Compression with Recurrent Neural Networks

**Authors:**    George Toderici, Damien Vincent, Nick Johnston, Sung Jin Hwang, David Minnen, Joel Shor, Michele Covell

---

[TOC]

---

## 0. Abstract

LSTM, associative LSTM

one-shot versus additive reconstruction architectures

new scaled-additive framework


4.3%~8.8% AUC improves comparing to previous work
(AUC: area under the rate-distortion curve)

## 1. Introduction



metrics:  PSNR-HVS and MS-SSIM

### 1.1 Previous Work

## 2. Methods


an encoding network E

a binarizer B

a decoding network D

E和D都是包含recurrent network components



a single iteration of our networks:

![CM28fI.png](https://s1.ax1x.com/2018/04/22/CM28fI.png)


D_t,E_t 表示在迭代t时的带有状态的decoder和encoder, b_t是逐步的binary representation


24 * 8 = 192

### 2.1 Types of Recurrent Units

**LSTM**

![CM2ccV.png](https://s1.ax1x.com/2018/04/22/CM2ccV.png)

x_t: input
c_t: cell
h_t: hidden states

W和U， 是convolutional linear transforms
U convolutions和W convolutions有相同的depths


**Associative LSTM**

![CM2bjK.png](https://s1.ax1x.com/2018/04/22/CM2bjK.png)

除了输入x_t, 输出h\~_t, 门值 f, i, o 是实数值, 剩下的数值都是复数。

bnd(z) = z 如果|z| <= 1 否则为 z / |z|.


和non-associative LSTM一样，W和U都是卷积线性变换.

**Gated Recurrent Units**

![CMRAHg.png](https://s1.ax1x.com/2018/04/22/CMRAHg.png)


hidden state/output ht



