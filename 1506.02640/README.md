download: [https://arxiv.org/pdf/1506.02640]

title: You Only Look Once: Unified, Real-Time Object Detection

authors: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

[TOC]

---

### Abstract

1 network predicts bounding box and class probabilities from full image.


end to end
fast: 45 fps(base YOLO), 155fps(fast YOLO)

YOLO makes more localization error but less to predict detections where nothing exists(FP rate low, 误检率较低)

generalization good from natural images to artwork on both the Picasso Dataset and the People-Art Dataset

### 1. Introduction

![CsKvqK.png](https://s1.ax1x.com/2018/05/15/CsKvqK.png)

1. YOLO is extremely fast
2. YOLO reasons globally about the image when making predictions
3. YOLO learns generalizable representations of objects

在accuracy方面, YOLO还落后于其他SOTA的detection system, 虽然YOLO可以迅速识别出objects, 但是精确定位objs还是有困难. 尤其是对于小物体的检测.

### 2. Unified Detection

1. 划分输入图像为SxS(**S=7 on PASCAL VOC**)的网格, 如果一个物体的gt的center落在了某个网格内,则该cell就负责检测出该物体。
2. 每一个网格cell都同时预测B(**B=2**)个bounding boxes和对应的confidence scores。 confidence定义为Pr(Object) * IoU(truth, pred), 如果没有物体, confidence score应该是0, 否则我们希望conf score应该等于IoU.

3. 每个bbox都包含5个预测: x,y,w,h和conf: (x,y)预测bbox的center coordinates relative to the bounds of the grid cell. w,h是相对于整个image.

每个grid cell也包含C(**C=20 bcz VOC has 20 labelled classes**)个条件概率: Pr(Class_i | Object), 不论#bbox B多大, 我们只预测一次class probabilities per grid cell.

测试时，我们将cond prob与conf相乘:

![CsQSO0.png](https://s1.ax1x.com/2018/05/15/CsQSO0.png)

![CsQCwT.png](https://s1.ax1x.com/2018/05/15/CsQCwT.png)

7x7x30 tensor(output)


#### 2.1 Network Design

inspired by the GoogLeNet model:

24 conv layers ==> 2 fc 

[![CsQZlR.md.png](https://s1.ax1x.com/2018/05/15/CsQZlR.md.png)](https://imgchr.com/i/CsQZlR)

instead of inception module, simply use 1x1 reduction layers followed by 3x3 conv layers.

fast version of YOLO: 9 conv 

#### 2.2 Training

pre-train on ImageNet 1000-class classification task

use the first 20 conv layers + avg pooling and a fully connected layer.

train for approximately a week and achieve a single crop top-5 acc 88% on ImageNet2012 val set.

We use the **Darknet** framework.

we add 4 conv layers and 2 fc layers with random initialized weights.

increase the input resolution from 224x224 to 448x448.

![CslEE8.png](https://s1.ax1x.com/2018/05/15/CslEE8.png)

lr schedule比较有趣: 前几个epochs，我们慢慢增大learning rate从10\^-3到10\^-2: 因为如果一开始学习率比较大的话，因为不稳定的梯度信息, model通常会diverges; 然后用10\^-2训练75个epochs, 然后10\^-3训练30个epochs, 最后用10\^-4训练30个epochs.

#### 2.3 Inference

Non-maximal suppression adds 2~3% in mAP.

#### 2.4 Limitations of YOLO

each grid cell only predict 2 boxes and can only have one class.

limits the number of nearby objects that our model can predict.

our model learns to predict bbox from data, it struggles to generalize to objects in new or unusual aspect ratios or configurations.

use relatively coarse features since our arch has multiple downsampling layers from the input image.



Our main source of error is incorrect localizations.

### 3. Comparison to Other Detection Systems

#### Deformable parts models

#### R-CNN

#### Other Fast Detectors

#### Deep MultiBox

#### OverFeat

#### MultiGrasp

### 4. Experiments

#### 4.1 Comparison to Other Real-Time Systems

#### 4.2 VOC 2007 Error Analysis

#### 4.3 Combining Fast R-CNN and YOLO

#### 4.4 VOC 2012 Results

#### 4.5 Generalizability: Person detection in Artwork

### 5. Real-Time Detection In The Wild

### 6. Conclusion



