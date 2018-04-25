**Download:**   [https://arxiv.org/pdf/1801.09392]

**Title:** Shift-Net: Image Inpainting via Deep Feature Rearrangement

**Authors:**  Zhaoyi Yan, Xiaoming Li, Mu Li, Wangmeng Zuo, Shiguang Shan

---

[TOC]

---


## 0. Abstract

context encoder


shift-connection layer to the U-Net, namely Shift-Net, filling in missing region of any shape with sharp structures and fine-detailed textures.



Paris Street View and Places

## 1. Introduction

## 2. Related Work

### 2.1 Exemplar-based inpainting
从外往内 ， 通过搜索和复制最符合的patches从known region
PatchMatch algorithm ==> generalize kNN

更适合于生成textures, 而不适合保留edges和structures.

Global image coherence

在大多数exemplar-based inpainting methods, missing parts is recovered as shift representation of the known region in pixel/region level.



### 2.2 CNN-based inpainting

context encoder
Semantic image inpainting
MNPS(Multi-scale neural patch synthesis)

### 2.3 Style transfer

## 3. Method

输入图像I
I\^gt 恢复目标
采用(U-Net)[Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI). (2015)]

shift-operation + guidance loss

### 3.1 Guidance loss on decoder feature

missing region 表示为 Ω
known region 表示为 Ω'

explicitly model the relationship between φ(L-l)(I)y and φ(l)(I\^gt)y
![Cl7XsH.png](https://s1.ax1x.com/2018/04/25/Cl7XsH.png)

对于任意的x∈Ω', ![Cl7xeA.png](https://s1.ax1x.com/2018/04/25/Cl7xeA.png)
因此, 这个guidance loss只定义在Ω上。

### 3.2 Shift operation and Shift-Net

![ClLkQO.png](https://s1.ax1x.com/2018/04/25/ClLkQO.png)

shift vector is ![ClLAyD.png](https://s1.ax1x.com/2018/04/25/ClLAyD.png)

定义:

![ClL2kR.png](https://s1.ax1x.com/2018/04/25/ClL2kR.png)

卷积层特征 φ(L-l)(I), φ(l)(I) 和 φ\^shift(L-l)(I)  连接在一起

shift-operation不同于exemplar-based inpainting, 从以下几个方面:

1. exemplar-based inpainting从pixels/patches操作, 而shift从deep encoder feature domain操作.
2. with the guidance of φ(L-l)(I), all the shift vectors can be computed in parallel.
3. In contrast, in shift operation φ(L-l)(I) is learned from large scale data and is more powerful in capturing global semantics. 
4. Adopt a data-driven manner to learn an appropriate model for image inpainting


### 3.3 Model objective and learning


**Objective**

φ(I;W)  W是参数
除了guidance loss, 还有l1和adversarial loss

l1 loss:

![ClO7vT.png](https://s1.ax1x.com/2018/04/25/ClO7vT.png)

adv loss:

![ClOqrF.png](https://s1.ax1x.com/2018/04/25/ClOqrF.png)

W就是Generator parameters

overall objective:

![ClOXVJ.png](https://s1.ax1x.com/2018/04/25/ClOXVJ.png)

**Learning**

training set {(I, I\^gt)}

shift-connection layer


![ClOzP1.png](https://s1.ax1x.com/2018/04/25/ClOzP1.png)

P置换矩阵, 每个元素都是{0,1}, 每行只有一个1.


[![ClOja9.md.png](https://s1.ax1x.com/2018/04/25/ClOja9.md.png)](https://imgchr.com/i/ClOja9)

add the shift-connection layer at the resolution of 32x32.


前两项和U-Net一样, 第三项的导数也可以直接求出，因此可以end-to-end训练。


## 4. Experiments

Experiment on 
* 2 datasets: PSV(paris street view) 和 Places365-Standard dataset中的6个场景.
* real world images


PSV: 14,900 training imgs and 100 test imgs.
Places365-Standard Dataset: 1.6 million training imgs from 365 scene categories.

被选出的6个场景为: butte, canyon, eld, synagogue, tundra and valley

each categories has 5,000 training imgs and 900 test imgs and 100 validation imgs.

resize to let its minimal length be 350, randomly crop a subimage of size 256*256

| optim  | lr  |β1|bs|stop epochs|data argument|λg|λadv|
|---|---|---|---|---|---|---|---|---|
| Adam  | 2x10\^-4  | 0.5|1|30|flipping|0.01|0.002|

### 4.1 Comparisons with SOTA

### 4.2 Inpainting of real world images

## 5. Ablative Studies

烧蚀研究

[TO-READ]
### 5.1 Effect of guidance loss

### 5.2 Effect of shift operation at different layers
### 5.3 Effect of the shifted feature

### 5.4 Comparison with random shift-connection

## 6. Conclusion

In future
* improve the speed of nearest searching in the shift operation
* introduce multiple shift-connection layers
* extend shift-operation to other low level vision tasks(compression?)



## Supplementary material

### A. Definition of masked region in feature maps

All the elements of the filters are 1/16,
[![CljEOU.md.png](https://s1.ax1x.com/2018/04/25/CljEOU.md.png)](https://imgchr.com/i/CljEOU)

### B. Details of Shift-Net

Table 2 and Table 3

instance normalization

encoder part of G is stacked with Conv-InstanceNorm-LeakyReLU
decoder part of G consists of seven deconv-IN-ReLU

*30x30x1 of D's output (real or fake) ?*
### C. More comparisons and object removals

...
