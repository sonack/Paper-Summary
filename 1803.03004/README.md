download:   [https://arxiv.org/pdf/1803.03004.pdf]
title:  Learning Effective Binary Visual Representations with Deep Networks
authors:    Jianxin Wu and Jian-Hao Luo,  Nanjing University

## Summary

提出了一种新的二值化特征的方法，传统方法一般是先用sigmoid将其限制在(0,1)，再近似为二值的0和1，或者是利用h(x)=tanh(αx)，当α很大时，h(x)就接近于sign函数。无论是sigmoid还是tanh，都是饱和激活函数，训练慢还有梯度消失问题。
本文提出了一种新的ABC(Approximately Binary Clamping)层，它是非饱和的，且可以产生true binary representations，其公式如下：

![](https://s1.ax1x.com/2018/03/19/9TPUYR.png)

[![9TPB6K.png](https://s1.ax1x.com/2018/03/19/9TPB6K.png)](https://imgchr.com/i/9TPB6K)



当斜率r为1时，类似于y=x，除了在x>0时上移了一个单位，当0<r<1时，梯度是非饱和的，当r=0时，就是hard的binary function。在训练时，让r逐渐趋向于0（比如，一共20个epoch，在第17个epoch上将r设置为0），则最终将得到true binary representation。然而r=0后，还可以继续训练后面的classifier。

为啥r->0时，|rx|->0呢？这是因为一般在ABC层前面加BN层，x的scale是稳定的，类似于无穷小乘以有界量还是无穷小的道理。**论文中说在ABC前面加BN是很关键的**

论文实现了Torch和Caffe的ABC层，并做了一系列实验。



scaled tanh:

[![9TPgkd.png](https://s1.ax1x.com/2018/03/19/9TPgkd.png)](https://imgchr.com/i/9TPgkd)

α>0，是一个scaling parameter，类似于ABC中的r.



缺点：

1. 永远不可能是真的binary，如果α很大，容易数值错误，造成nan;
2. soft binary，减慢了收敛速度，梯度几乎为0.



ABC层的优点在于，其梯度一直为r，不会成为梯度传递的阻碍。有个小问题就是如果r衰减为原来的1/k，则ABC层后面的梯度虽然不受影响，但ABC层前面的层的梯度都将变为原来的1/k，但我们在实际训练时本来learning rate就是应该不断衰减的，因此可以通过以下trick来解决：如果常规训练时，我们衰减learning rate k次，那么我们在使用ABC时就衰减r和lr各sqrt(k)次





## Experiments

3种不同任务：

1. 学习short binary codes来做Image retrieval
2. 学习long binary visual representations来做large scale image recognition(+ 收敛性研究)
3. 泛化学到的recognition network到object detection



### image retrieval

与adaptive tanh作比较，数据集为CIFAR10和NUS-WIDE，基本的CNN架构为DSH中的结构。

signum function: 符号函数。

### Large scale recognition
Imagenet image recognition

fine-tune Facebook的ResNet-50，每4个epoch，lr和r都乘以sqrt(0.1)，但是最后4个epochs(17~20),直接将r设置为0.

Facebook Torch ResNet-50 + ABC  v.s.    Caffe ResNet-50 + adaptive tanh

official differences:
Facebook:   top1 error 24.018%
ABC:    24.316%         下降了只有0.298%
Caffe:      top1 error 24.365%
Adaptive tanh:  26.430%     下降了2.065%

### Generalizing to object detection

ABC-2048
ABC-4096

ABC + Fast R-CNN

VOC07 trainval dataset

corroborate vt.证实了，使坚固


**The feature scale(magnitude) are not as important as signs(activated or not).**

![9TAzG9.png](https://s1.ax1x.com/2018/03/19/9TAzG9.png)

0.22和6.78都是表示边界，大部分是背景，小部分是目标，但是差异很大，说明scale的大小并没有那么明确的意义，重要的是activated or not(大于0还是小于0).