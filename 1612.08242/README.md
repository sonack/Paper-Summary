download: [https://arxiv.org/pdf/1612.08242]

title:
YOLO9000: Better, Faster, Stronger

authors: Joseph Redmon, Ali Farhadi

[TOC]

---

Why named YOLO9000? can detect over 9000 object categories.

### 1. Introduction

harness the large amount of classification data we already have and use it to expand the scope of current detection systems.

YOLO -> YOLO v2 -(dataset combination method and joint training algo to train)-> YOLO9000(a model on more than 9000 classes from ImageNet and detection from COCO).

### 2. Better

**Batch Normalization**

bn on all conv layers --> mAP ↑ 2% & remove dropout

**High Resolution Classifier**

AlexNet类似的CNN的输入大小一般都小于256x256.

原始的YOLO在224x224上训练, 然后为了detection而增加到448。 这意味着网络必须同时切换到学习object detection和调整到新的输入resolution两个任务.
对于YOLO v2, 我们首先在ImageNet上用完整的448x448 ft 分类网络 10个epochs, 先让网络适应好在更高的resolution input上工作的更好，再ft 结果网络来detection. 这个高resolution分类网络提高了4%的mAP.

**Convolutional With Anchor Boxes**

YOLO直接使用fc来预测bbox坐标.

预测offsets而不是直接预测坐标简化了问题, 更容易使网络学习.

remove fc from YOLO, use anchor boxes to predict bbox:

1. eliminate one pooling layer;
2. shrink the network to operate on 416 input images instead of 448(奇数, 以便有a single center cell);
3. decouple the class prediction mechanism from the spatial location;

98 boxes per image --> more than a thousand boxes

acc ↓   recall ↑


**Dimension Clusters**

anchor box的维度 手工选择?

run k-means clustering on the training set bounding boxes to automatically find good priors.

d(box, centroid) = 1 - IoU(box, centroid)

k=5: good tradeoff

与hand-picked的anchor boxes很不同: 有更少的短宽boxes 有更多的高瘦boxes;


**Direct localtion prediction**

使用anchor box的原始的预测(x,y)不稳定:

x = (t_x * w_a) - x_a
y = (t_y * h_a) - y_a

这种formulation是不受限制的, 因此最终anchor box能够落在图片的任何点处, 不论预测它们的本来位置在哪里。

所以我们不预测offset, 而采用YOLO的做法，直接预测相对于location of the grid cell的坐标位置. 这就限制了gt只能落在0到1之间(使用了logistic activation来限制)。

the network predicts 5 bbox at each cell in the output feature map.
每个bbox预测5个坐标: t_x, t_y, t_w, t_h, t_o.


如果the cell距离图片左上角的偏移量是(c_x, c_y), bbox先验的宽高为p_w, p_h, 则预测对应为:

![CcYAkn.png](https://s1.ax1x.com/2018/05/19/CcYAkn.png)

![CcYH3V.png](https://s1.ax1x.com/2018/05/19/CcYH3V.png)


使用dimension clusters和directly predict the bbox center location  mAP ↑ 5% over the version with anchor boxes.


**Fine-Grained Features**

Simply add a passthrough layer that brings features from an earlier layer at 26x26 resolution.

The passthrough concatenate the higher resol feat with low resol feat by stacking adaj feat into different channels instead of spatial locations.

将26x26x512的feat map变成了13x13x2048 然后可以和原始feat相concat.

1% performance ↑


**Multi-Scale Training**

224x224 -> 448x448(ease the adaptation of higher resolution) -> 416x416(anchor box)

我们的网络是全卷积, 可以适应任何大小的输入.

每10个batches, 网络随机选择一个新的图像尺寸, 32的倍数, {320, 352, ... , 608} [10~19], 通过改变网络的size, 可以简单的trade-off YOLOv2的speed和accuracy.

low resol YOLOv2: 228x228  90FPS
high resol YOLOv2: VOC07 mAP 78.6

**Further Experiments**

[![CcYxE9.md.png](https://s1.ax1x.com/2018/05/19/CcYxE9.md.png)](https://imgchr.com/i/CcYxE9)

### 3. Faster

VGG-16 as base feature extractor too needlessly complex:
conv layers of VGG-16 require 30.69 billion floating point operations for a single pass over a single image at 224x224 resolution.

YOLO use a custom network based on Googlenet arch to replace VGG-16:  8.52 billion operations for a forward pass.  88% vs 90%

**Darknet-19**

3x3 filters and double the number of channels after every pooling step

global average pooling (following Network in Network) to make predictions as well as 1x1 filters to compress the feature representation between 3x3 convs.

batch normalization to stabilize training, speed up convergence and regularize the model.



Darknet-19: 19 conv layers + 5 maxpooling layers

5.58 billion operations , 72.9% top-1 acc and 91.2% top-5 acc on ImageNet.


**Training for classification**

standard ImageNet 1000 class classification for 160 epochs using SGD with a init lr = 0.01, polynomial lr decay with a power of 4, weight decay of 0.0005, momentum 0.9 using the Darknet neural network framework.

after our initial training on images at 224x224, we ft our network at a larger size, 448. with above params but for only 10 epochs and starting at inital lr of 10^-3. 经过这个high resolution, 我们的网络达到了top-1 acc为76.5%, top-5 acc为93.3%.


**Training for detection**

修改网络结构: 移除最后一层的conv layer, 增加3个3x3 conv layers with 1024 filters each followed by a final 1x1 conv layer with the number of outputs we need for detection.

对于VOC来说, 预测5个boxes(k=5)的5个coordinates each and 20 classes per box so 125 filters.


add a passthrough from the final 3x3x512 layer to the second to last conv layer so that our model can use fine grain features.


train the network for 160 epochs, starting lr=10^-3, dividing it by 10 at 60 and 90 epochs.

we use a weight decay of 0.0005 and momentum of 0.9.


### 4. Stronger

提出了一种jointly-training on classification and detection data的机制.


训练时， 混合Images from both detection and classification datasets.
当网络看到一张图片是labelled for detection, we can backpropagate based on the full YOLOv2 loss function, 当看到是一张用于分类任务的图像时，我们只反向传播用于分类的部分loss.

问题: 检测的label一般比较粗粒度, 如dog, 而分类的label比较具体, 比如Norfolk terrier\Yorkshire terrier. 如果想训练, 需要找一种merge these labels的方法。

如果用softmax, 则假设这些classes是互斥的. 因此我们使用一种 multi-label model , 它不假设互斥.

**Hierarchical classification**

ImageNet的labels是来自于WordNet, 一个language database that structure concepts and how they relate.

WordNet被组织成一个有向图, 而不是一个tree, 因为语言很复杂, 比如"dog"既是属于canine(犬科), 又属于domestic animal(驯养动物), 为了简化, 我们只使用hierarchical tree from the concepts in ImageNet.

path to physical object
many synsets only have one path through the graph, we add all of those paths to our tree first.

iteratively examine the concepts left and add the paths that grow tree as little as possible.

form the final 'WordTree'.

我们在每一个节点预测每一个下位词(hyponym)的条件概率, 例如对于"terrior"节点，预测:

Pr(Norfolk terrier | terrier)
Pr(Yolkshire terrier | terrier)
Pr(Bedlingto terrier | terrier)
...

WordTree1k , we add all of the intermediate nodes which expands the label space from 1000 to 1369.


对于分类任务, 假设根节点Pr(physical object) = 1
对于检测任务, Pr(physical object) = objectness predictor

detector predicts的是bounding box and the tree of probabilities, 从根节点down traverse the tree, 每次分叉都选择最大概率的分支, 直到达到某些阈值, 然后我们就预测为这类object.

**Dataset combination with WordTree**

simply map the categories in the datasets

**Joint classification and detection**

top 9000 classes ==> 9418 classes  4:1 (ImageNet:COCO)


k = 3 rather than 5
highest probability for that class (classification image)
objectness loss: .3 IOU

44 object categories shared with COCO
YOLO9000  19.7mAP overall  16.0mAP(on the disjoint 156 classes)

### 5. Conclusion

It can be run at a variety of image size to provide a smooth tradeoff between speed and accuracy.


