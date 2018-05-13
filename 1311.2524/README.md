download: [https://arxiv.org/pdf/1311.2524]

title: 
Rich feature hierarchies for accurate object detection and semantic segmentation

authors: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik

[TOC]

---



## Appendix

### B


为什么用来fine-tune CNN和训练SVM的正例和反例不同?

在ft时, 我们将每一个proposal对应到gt instance, 该gt instance有最大的IoU重叠, 如果IoU至少0.5，则将其label为对应的gt class为positive。所有其他的proposals都被Label为"background"(i.e. 对所有类都是negative examples);

在训练SVM时, 我们只将gt boxes作为positive examples, 标记proposals为某class的negative examples(如果该proposal与该class的任何instances的IoU都小于0.3的话), 其他的非gt, 但是>0.3IoU的proposals都被忽略。


为什么ft后还要再训练SVM呢?

mAP会掉， 可能是多因素:
1. 无法精确localization(因为ft的positive example和negative example的定义)
2. 没有像训练SVM一样用'hard negative examples'

可能不用训练SVMs, 也能弥补这种gap,以后研究.

### C Bounding-box regression


class-specific bounding-box regressor

we regress from features computed by the CNN.

regression algo:

input:  a set of N training pairs {(Pi, Gi)}i=1...N, 其中Pi是一个四元组, 用来指定proposal Pi的bounding box的中心位置和宽高。 Pi = {Pi_x, Pi_y, Pi_w, Pi_h}. 每一个gt bounding box G同样方式来描述: G = {G_x, G_y, G_w, G_h}. 我们希望学到一个transformation可以从一个proposed box P 映射到 gt box G.

参数化该transformation为4个函数: d_x(P), d_y(P), d_w(P), d_h(P).

前两个指定P的bounding box的scale-invariant translation of the center of P's bounding box, 后两个则在log-space内指定translations of the width and height of P's bounding box.

通过应用以下变换， 我们可以将一个输入的proposal P 转换为 一个预测的 ground-truth box G':

![CD1VMt.png](https://s1.ax1x.com/2018/05/13/CD1VMt.png)



