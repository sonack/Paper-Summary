download:   [https://arxiv.org/pdf/1803.06131.pdf]

title:  Towards Image Understanding from Deep Compression without Decoding

authors:    Robert Torfason, Fabian Mentzer, Eirikur Agustsson, Michael Tschannen, Radu Timofte, Luc Van Gool



## Abstract

直接把图像理解任务（如分类和分割）与DNN-based图像压缩方法产生的compressed representation上相结合，这样就可以不用解压到RGB空间再做相关任务，节省了计算复杂量2x，但是识别任务的准确度却和直接在RGB空间相当。进一步，可以joint train 压缩网络和分类网络(on the compressed representation)，两者协同，提升了图像质量，分类准确度和分割性能。研究发现，在低码率下，直接在compressed representations上做推断要比在compressed RGB images上做更有利。

## Introduction

learned compression algorithms v.s engineered compression algorithms

learned compression algos:

1. easily be adapted to specific target domains
2. amenability to learning and inference without reconstruction.


![9qIe3Q.png](https://s1.ax1x.com/2018/03/25/9qIe3Q.png)

灰色代表(E,D)，白色代表推断网络。

这篇文章的贡献：

1. 在压缩域上做了2个不同的任务，图像分类和语义分割
2. 发现在编码域做分类的精确度和在解压图上做分类几乎一样，然而却需要1.5x~2x更少的操作。
3. 发现在编码域上做语义分割的效果和在解压图域（在中等压缩率下）上做分割的效果相当，在低码率下的效果反而更好。
4. 当joint-training时，SSIM和分类准确度都变好了
5. 我们的方法只需要少量修改。

## Related Works

大家很久之前就在尝试在编码域（engineered codecs）上直接进行任务了，然而，这是在learned image compression algorithms产生的compressed representations上第一次做inference.

## Learned Deeply Compressed Representation

### Deep Compression Architecture

使用了Theis et al提出的convolutional autoendoer作为compression network，使用scalar quantization.
参考Appendix A.1.

产生的特征图大小为(w/8, h/8, C)

D+βR

D使用MSE, R使用H(q)估计，利用直方图估计。使用β来控制MSE和entropy的trade-off，对每个β，获得了一个操作点，图像在某个bpp，和对应的MSE。

获得了3个操作点：

* 0.0983bpp
* 0.330bpp
* 0.635bpp

*bpp: 一个操作点的bpp是通过validation set上所有图片的bpp平均得到的。*

Appendix A.2 展示了这些操作点上与JPEG和JPEG2000在压缩指标上的比较。



## Image Classification From Compressed Representations

### ResNet For RGB Images

使用ResNet50(V1)来分类RGB空间上的图片，它是由所谓的bottleneck residual units组成的，每个unit的计算代价都是一样的（无关输入tensor的空间尺寸）【下采样的blocks和root-block除外】，它是全卷积网络，输入空间尺寸为224*224.

调整14x14(conv4_x)块的数量获得了ResNet-71，是ResNet-50和ResNet-101的中间物。

### ResNet For Compressed Representations

224x224 -> 28x28xC

ResNet的简单变体，使用compressed representation作为输入，叫做cResNet-k，c表示"compressed representation", k是网络中卷积层的数量。

简单来说，就是切断ResNet的前部：简单移除了root-block和residual layers that have a larger spatial dimension than 28 * 28.

最终得到了3种不同架构：



i. cResNet-39	(ResNet-50的前11层被removed)

ii. cResNet-51

iii. cResNet-72 (添加14x14 residual blocks以匹配ResNet-50和ResNet71的计算复杂度)



![9qIz5T.png](https://s1.ax1x.com/2018/03/25/9qIz5T.png)





### Benchmark

数据集:	ILSVRC12  训练classification networks和compression network

1.28 million training images

50k	validation images

图像分类：报告top1和top5准确度，使用val set的224x224 center crops (RGB images)和28x28 center crops(compressed representation)



### Training Procedure

给定trained compression network, 首先fix住压缩网络，训练分类网络（无论是RGB空间还是压缩域）

压缩域：将fixed的encoder的output输入到cResNets中

RGB域：将fixed的encoder-decoder的输出(RGB images)送入ResNets中



Appendix A.4详细描述。



### Classification Results



![9qoeIK.png](https://s1.ax1x.com/2018/03/25/9qoeIK.png)



蓝的(ResNet-50)和黄的(cResNet-51)在复杂度上是相近的，他们的性能也相近(ResNet-50稍微好一点)。

当bpp减少时，两者的差距越来越小。



如果把原图RGB送入ResNet-50，top-5 accuracy是89.96%。





## Semantic Segmentation From Compressed Representations

### Deep Method

ResNet-based Deeplab architecture

https://github.com/DrSleep/tensorflow-deeplab-resnet

atrous convolutions:  filters are upsampled instead of downsampling the feature maps

参考： http://arxiv.org/abs/1606.00915

最后的1000-way classification layer 被ASSP(Atrous spatial pyramid pooling)替换，有4个并行分支{6,12,18,24}，用来提供最后的逐像素分类。



### Benchmark

Dataset: PASCAL VOC-2012

20 object foreground classes and 1 background class

1464 training

1449 validation

after futhermore augmented with extra annotations , the final dataset has 10,582 images for training and 1449 images for validation

Performance measure:

pixelwise intersection-over-union (IoU) averaged over all the classes

mean-intersection-over-union(mIoU) on the validation set



### Training Procedure

cResNet/ResNet are pre-trained on the classification task

the (E,D) are fixed, and the architectures are finetuned on the semantic segmentation task.

然后用dilated convolutions来适应， cResNet-d / ResNet-d

training procedure和Chen et al.所用的设置相同，除了pre-processing过程有点不同，参见Appendix A.5



### Segmentation Results

![9q7dDs.png](https://s1.ax1x.com/2018/03/25/9q7dDs.png)

不同于classification的是，对于semantic segmentation，ResNet50和cResNet51在0.635bpp时表现的同样好。

0.330bpp，segment from compressed representations表现的稍微更好一点，0.0983bpp时，差距更明显。



## Joint Training For Compression And Image Classification

### Formulation

combined loss function:

![9q7Wr9.png](https://s1.ax1x.com/2018/03/25/9q7Wr9.png)



1. Initialized the compression network and the classification network from a trained state obtained before. Then both finetune jointly.
2. See Appendix A.8

为了确保分类准确度的上升不是仅仅由于

1. 更好的压缩率操作点
2. 更长的训练时间

我们这样做：

我们通过finetune后获得了一个新的操作点，然后再train一个cResNet51在其上(from scratch)，最终，固定住这个压缩网络，我们同样训练这个cResNet-51同样个epochs，这样的话，这个模型就可以用来比较了。

对于segmentation，只有pre-trained网络的不同。



### Joint Training Results

![9qHVZn.png](https://s1.ax1x.com/2018/03/25/9qHVZn.png)

1. joint training并没有影响compression performance significantly   (MS-SSIM和SSIM增加了一点小， PSNR减少了一点)
2. 对于cResNet-51，经过ft(finetune)，classification和segmantation的性能指标都有所上升。

## Discussion



主要优势在于：

* Runtime
* Memory
* Robustness
* Synergy(协同作用)
* Performance



缺点:

* Complexity
* Performance





## Appendix

[TODO]