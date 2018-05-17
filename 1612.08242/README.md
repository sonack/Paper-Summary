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

the network predicts 5 bbox at each cell in the output feature map.
每个bbox预测5个坐标: t_x, t_y, t_w, t_h, t_o.




