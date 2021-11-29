# Neighbourhood-Component-Analysis

近邻成分分析（Neighbourhood Component Analysis，NCA）是由Jacob Goldberger和Geoff Hinton等大佬们在2005年发表的一项工作，属于度量学习（Metric Learning）和降维（Dimension Reduction）领域。其关键点可以概括为：任务是KNN Classification，样本相似度计算方法基于马氏距离（Mahalanobis Distance），参数选择方法为留一验证法（Leave One Out）。最后模型可以学习样本的低维嵌入表示（Embedding），既属于度量学习范畴，又是降维的过程。

近邻成分分析在度量学习和降维领域展现了很强大的本领，也为后来很多降维的工作起了引导性的作用。本文将详细介绍NCA的思想和数学形式，然后也会围绕如何快速实现NCA进行介绍，利用Python3.6实现，最后给出一些自己实验得到的结果，以便更加了解NCA的能力与缺陷。

下面是本文的主要内容：

1. NCA的主要思想、数学形式及求解
2. NCA的代码实现：四种方法
3. NCA在一些数据集上的表现

## **1 NCA的主要思想、数学形式及求解**

## 1.1 NCA主要思想

首先，NCA中的Neighbourhood是指近邻，可以理解为相似的样本（这里默认距离小代表相似度高）。在降维和度量学习任务中，“邻居”是一个很重要的思想，很多算法都是围绕它进行推导的。NCA基于KNN Classification，所以，NCA是一个有监督的算法，必须给定数据和类别标签才可以进行训练。

如果想理解NCA，必须要先理解NCA的三个关键点，分别是：Stochastic KNN、Metric Learning和Leave one out。

- NCA是有监督的，基于分类器Stochastic KNN；
- NCA衡量近邻相似度的方法借助了Metric Learning；
- NCA在训练过程中调参采用的是Leave one out验证法；

注意，虽然NCA借助使用了分类器KNN，但是NCA并不是做分类问题的，而是度量学习和降维。下面就用数学公式来形式化表述NCA做的事情。

## 1.2 NCA数学形式

给定数据样本 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%28x_1%2Cy_1%29%2C%28x_2%2Cy_2%29%2C%5Ccdots%2C%28x_n%2Cy_n%29%5C%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=x_i+%5Cin+R%5Ed) 代表第 i 个数据样本， ![[公式]](https://www.zhihu.com/equation?tex=y_i+%5Cin+R) 代表样本标签。考虑在KNN中，使用Leave one out交叉验证法，假设现在要预测第 i 个样本的标签，那么我们可以这么做：

1. 计算样本 i 和其余所有样本之间的欧氏距离， ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bij%7D+%3D+%7C%7Cx_i+-+x_j%7C%7C_2)
2. 选择距离最小的 k 个， ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bij_1%7D%2C+d_%7Bij_2%7D%2C%5Ccdots%2Cd_%7Bij_k%7D)
3. 利用这 k 个样本的标签进行投票得到预测结果， ![[公式]](https://www.zhihu.com/equation?tex=Vote%28y_%7Bj_1%7D%2C+y_%7Bj_2%7D%2C+%5Ccdots%2C+y_%7Bj_k%7D%29)



上述过程是一般的KNN过程，那么这里先引入Stochastic 1-NN改进办法：

1. 计算样本 i 的近邻分布：

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bij%7D+%3D+%5Cfrac%7B%5Cexp%5Cleft%28+-%7C%7Cx_i+-+x_j%7C%7C_2%5E2+%5Cright%29%7D%7B%5Csum_%7Bk%5Cneq+i%7D%5Cexp%5Cleft%28+-%7C%7Cx_i+-+x_k%7C%7C_2%5E2+%5Cright%29%7D%2C+p_%7Bii%7D+%3D+0+%5C%5C)

\2. 根据概率分布 ![[公式]](https://www.zhihu.com/equation?tex=p_%7Bij%7D%2C+j+%5Cneq+i%2C+j+%5Cin+%5B1%2Cn%5D) 采样得到一个样本 k，然后将第 i 个数据的样本预测为 ![[公式]](https://www.zhihu.com/equation?tex=y_k)



从上面可以看出，第 i 个样本的真实标记为 ![[公式]](https://www.zhihu.com/equation?tex=y_i) ，假如 ![[公式]](https://www.zhihu.com/equation?tex=y_k+%3D+y_i) ，那么预测正确。记 ![[公式]](https://www.zhihu.com/equation?tex=C_i+%3D+%5C%7B+j+%7C+y_j+%3D+y_i%5C%7D) 表示与第 i 个样本类别一样的下标集合，那么利用上述 Stochastic 1-NN正确预测第 i 个样本标签的概率为：

![[公式]](https://www.zhihu.com/equation?tex=p_i+%3D+%5Csum_%7Bj+%5Cin+C_i%7D+p_%7Bij%7D+%5C%5C)
那么，对于所有的样本，优化目标为：

![[公式]](https://www.zhihu.com/equation?tex=f+%3D+%5Csum_%7Bi%3D1%7D%5En+p_i+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Csum_%7Bj+%5Cin+C_i%7D+p_%7Bij%7D+%5C%5C)
到这里，不免产生一个问题，Stochastic 1-NN里面没有什么参数可以学习，给定样本集合最终的优化目标 ![[公式]](https://www.zhihu.com/equation?tex=f) 就是一个确定值，目前为止还没真正构成一个优化问题。再鉴于使用欧氏距离计算会导致计算量特别大，并且维度空间特别高，这里再引入度量学习的思想，引入可学习的马氏距离。在进一步介绍之前，先看一下马氏距离的定义以及度量学习的定义。

## 1.3 马氏距离和度量学习

数据样本矩阵表示为 ![[公式]](https://www.zhihu.com/equation?tex=X+%3D+%5Bx_1%3Bx_2%3B%5Ccdots%3Bx_n%5D%5ET)，这是以样本角度表示的，还可以以特征角度表示为 ![[公式]](https://www.zhihu.com/equation?tex=X+%3D+%5Bf_1%3Bf_2%3B%5Ccdots%3Bf_d%5D) 。假设样本间的协方差矩阵为 ![[公式]](https://www.zhihu.com/equation?tex=S+%5Cin+R%5E%7Bd%5Ctimes+d%7D) ，那么有：

![[公式]](https://www.zhihu.com/equation?tex=S_%7Bij%7D+%3D+Cov%28i%2C+j%29+%3D+%5Cfrac%7B1%7D%7Bn%7D+%5Cleft%28f_i+-+mean%28f_i%29%5Cright%29%5ET%28f_j+-+mean%28f_j%29%29+%5C%5C)

马氏距离的定义为：

![[公式]](https://www.zhihu.com/equation?tex=d%28x_i%2C+x_j%29+%3D+%5Csqrt%7B%28x_i+-+x_j%29%5ETS%5E%7B-1%7D%28x_i+-+x_j%29%7D+%5C%5C)

这里简单介绍一下马氏距离的优点：

- 相当于是对数据中心化和标准化，数据中心化和标准化后的马氏距离与原来的马氏距离一致
- 由于是相当于中心化和标准化，所以不受特征单位的影响
- 可以推导出可学习的马氏距离，是度量学习的基础
- 考虑样本总体特性，一般来说，两个样本放入不同的总体，计算得到的马氏距离不相等

度量学习是基于可学习的马氏距离，也称伪（pseudo）马氏距离：

![[公式]](https://www.zhihu.com/equation?tex=d_M%28x_i%2C+x_j%29+%3D+%5Csqrt%7B%28x_i+-+x_j%29%5ETM%28x_i+-+x_j%29%7D%2C+%5C%2C%5C%2C%5C%2C%5C%2C+M+%5Cin+S_%2B%5Ed%5C%5C)
其中 ![[公式]](https://www.zhihu.com/equation?tex=M) 是半正定矩阵（Positive Semi-Definite, PSD），是可以学习的参数，称为度量。度量学习的目标就是通过一些约束（比如，must-link pair和must-not-link pair）进行优化矩阵 ![[公式]](https://www.zhihu.com/equation?tex=M) ，从而学习到一个距离度量。

由于 ![[公式]](https://www.zhihu.com/equation?tex=M) 是半正定矩阵，那么存在 ![[公式]](https://www.zhihu.com/equation?tex=A+%5Cin+R%5E%7Bk+%5Ctimes+d%7D%2C+k+%3C+d%2C+k+%5Cgeq+rank%28M%29) ，满足 ![[公式]](https://www.zhihu.com/equation?tex=M+%3D+A%5ETA) ，那么：

![[公式]](https://www.zhihu.com/equation?tex=d_M%28x_i%2C+x_j%29+%3D+%5Csqrt%7B%28x_i+-+x_j%29%5ETA%5ETA%28x_i+-+x_j%29%7D+%3D+%7C%7CAx_i+-+Ax_j+%7C%7C_2+%5C%5C)

可以看出马氏距离做的事情，可以理解为先把数据降维到低维空间，然后再求欧氏距离。

## 1.4 NCA完整表达

下面回到NCA，通过引入距离度量得到下面的过程：

1. 令 ![[公式]](https://www.zhihu.com/equation?tex=A+%5Cin+R%5E%7Bk+%5Ctimes+d%7D) 为参数
2. 计算样本 i 的近邻分布：

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bij%7D+%3D+%5Cfrac%7B%5Cexp%5Cleft%28+-%7C%7CAx_i+-+Ax_j%7C%7C_2%5E2+%5Cright%29%7D%7B%5Csum_%7Bk%5Cneq+i%7D%5Cexp%5Cleft%28+-%7C%7CAx_i+-+Ax_k%7C%7C_2%5E2+%5Cright%29%7D%2C+p_%7Bii%7D+%3D+0+%5C%5C)

\3. 优化目标： ![[公式]](https://www.zhihu.com/equation?tex=f%28A%29+%3D+%5Csum_%7Bi%3D1%7D%5En+p_i+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Csum_%7Bj+%5Cin+C_i%7D+p_%7Bij%7D+%5C%5C)

到这儿，就介绍完了NCA的主要数学形式，再总结一下：

- NCA是有监督的，需要提供样本标签
- 借鉴度量学习引入距离度量 ![[公式]](https://www.zhihu.com/equation?tex=A) 来计算样本间距离
- 目标是使得Stochastic 1-NN的准确率最高
- 参数学习过程使用了Leave one out交叉验证的方法

## 1.5 NCA求解

那么下面的问题就是如何求解NCA？论文中直接使用梯度下降法，但由于目标不是凸的，所以只能得到局部最优解。至于目标为什么不是凸的，是因为：将马氏距离下的度量学习转换为基于样本投影矩阵的度量学习，会造成问题非凸。

下面来推导一下NCA优化目标的梯度，记 ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bij%7D+%3D+%5Cexp%5Cleft%28+-%5CVert+Ax_i+-+Ax_j+%5CVert%5E2+%5Cright%29) ，那么有：

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bij%7D+%3D+%5Cfrac%7Bg_%7Bij%7D%7D%7B%5Csum_%7Bk%5Cneq+i%7Dg_%7Bik%7D%7D+%5C%5C)

由：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+x%5ETD%5ETDx%7D%7B%5Cpartial+D%7D+%3D+2Dxx%5ET+%5C%5C)

得到：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+g_%7Bij%7D%7D%7B%5Cpartial+A%7D+%3D+-2g_%7Bij%7DA%5Cleft%28+x_i+-+x_j+%5Cright%29%5Cleft%28+x_i+-+x_j+%5Cright%29%5ET+%5C%5C)

那么：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p_%7Bij%7D%7D%7B%5Cpartial+A%7D+%3D+%5Cfrac%7B1%7D%7B%5Cleft%28+%5Csum_%7Bk+%5Cneq+i%7D+g_%7Bik%7D%5Cright%29%5E2%7D%5Cleft%28+%5Cfrac%7B%5Cpartial+g_%7Bij%7D%7D%7B%5Cpartial+A%7D+%5Csum_%7Bk+%5Cneq+i%7D+g_%7Bik%7D+-+g_%7Bij%7D+%5Csum_%7Bk+%5Cneq+i%7D%5Cfrac%7B%5Cpartial+g_%7Bik%7D%7D%7B%5Cpartial+A%7D+%5Cright%29+%5C%5C)

前半部分为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Cleft%28+%5Csum_%7Bk+%5Cneq+i%7D+g_%7Bik%7D%5Cright%29%5E2%7D+%5Cfrac%7B%5Cpartial+g_%7Bij%7D%7D%7B%5Cpartial+A%7D+%5Csum_%7Bk+%5Cneq+i%7D+g_%7Bik%7D+%3D+%5Cfrac%7B-2g_%7Bij%7DA%5Cleft%28+x_i+-+x_j+%5Cright%29%5Cleft%28+x_i+-+x_j+%5Cright%29%5ET%7D%7B%5Csum_%7Bk+%5Cneq+i%7D+g_%7Bik%7D%7D+%3D+-2p_%7Bij%7DA%5Cleft%28+x_i+-+x_j+%5Cright%29%5Cleft%28+x_i+-+x_j+%5Cright%29%5ET+%5C%5C) 后半部分为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Cleft%28+%5Csum_%7Bk+%5Cneq+i%7D+g_%7Bik%7D%5Cright%29%5E2%7Dg_%7Bij%7D+%5Csum_%7Bk+%5Cneq+i%7D%5Cfrac%7B%5Cpartial+g_%7Bik%7D%7D%7B%5Cpartial+A%7D+%3D++p_%7Bij%7D%5Cfrac%7B-2%5Csum_%7Bk%5Cneq+i%7Dg_%7Bik%7DA%5Cleft%28+x_i+-+x_k+%5Cright%29%5Cleft%28+x_i+-+x_k+%5Cright%29%5ET%7D%7B%5Csum_%7Bs+%5Cneq+i%7D+g_%7Bis%7D%7D+%5C%5C+%3D+-2p_%7Bij%7DA%5Cleft%28%5Csum_%7Bk%5Cneq+i%7Dp_%7Bik%7D%5Cleft%28+x_i+-+x_k+%5Cright%29%5Cleft%28+x_i+-+x_k+%5Cright%29%5ET%5Cright%29+%5C%5C)
所以 ：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p_%7Bij%7D%7D%7B%5Cpartial+A%7D+%3D+-2p_%7Bij%7DA%5Cleft%28+%5Cleft%28+x_i+-+x_j+%5Cright%29%5Cleft%28+x_i+-+x_j+%5Cright%29%5ET+-+%5Csum_%7Bk%5Cneq+i%7Dp_%7Bik%7D+%5Cleft%28x_i+-+x_k+%5Cright%29%5Cleft%28+x_i+-+x_k+%5Cright%29%5ET%5Cright%29%5C%5C)

目标梯度为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Csum_%7Bj+%5Cin+C_i%7D+%5Cfrac%7B%5Cpartial+p_%7Bij%7D%7D%7B%5Cpartial+A%7D+%3D+-2A%5Csum_i%5Csum_%7Bj%5Cin+C_i%7D+p_%7Bij%7D+%5Cleft%28%28x_i-x_j%29%28x_i-x_j%29%5ET+-+%5Csum_%7Bk%5Cneq+i%7Dp_%7Bik%7D%28x_i-x_k%29%28x_i-x_k%29%5ET+%5Cright%29+%5C%5C) 化简为，这里记下面的梯度表达为“梯度#1”：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+2A%5Csum_i+%5Cleft%28p_i%5Csum_%7Bk%5Cneq+i%7Dp_%7Bik%7D%28x_i-x_k%29%28x_i-x_k%29%5ET++-+%5Csum_%7Bj%5Cin+C_i%7D+p_%7Bij%7D%28x_i-x_j%29%28x_i-x_j%29%5ET%5Cright%29+%5C%5C)

求出梯度之后，利用梯度下降法优化即可得到NCA的训练结果。训练得到的映射矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A) 可以用来对数据进行降维，以便于降低数据维度。

这里再稍微提一下，对上面的梯度进行变形，以便于后续代码实现，再进一步化简：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+2%5Csum_i+%5Csum_%7Bj%7D+%5Cleft%28p_ip_%7Bij%7D+-++pmask_%7Bij%7D%5Cright%29+%28Ax_i-Ax_j%29%28x_i-x_j%29%5ET+%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=pmask_%7Bij%7D+%3D+p_%7Bij%7D%2C+if+%5C%2C+j+%5Cin+C_i%2C+else%5C%2C+0) 。记 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bij%7D%3D+p_ip_%7Bij%7D+-+pmask_%7Bij%7D) ，有：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+2%5Csum_i+%5Csum_%7Bj%7D+W_%7Bij%7D+%28Ax_i-Ax_j%29%28x_i-x_j%29%5ET++%3D+2%5Csum_i+%5Csum_%7Bj%7D+W_%7Bij%7D+%28Ax_ix_i%5ET+%2B+Ax_jx_j%5ET+-+Ax_ix_j%5ET+-+Ax_jx_i%5ET%29+%5C%5C+%3D+2%28XA%5ET%29%5ET+%5Cleft%28+diag%5Cleft%28sum%28W%2C+axis+%3D+0%29%5Cright%29+%2B+diag%5Cleft%28sum%28W%5ET%2C+axis+%3D+0%29%5Cright%29+-+W+-+W%5ET+%5Cright%29+X+) 上式第三个等号是因为：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%7D%5Csum_%7Bj%7DW_%7Bij%7D+x_ix_j%5ET+%3D+X%5ETWX+%5C%5C+%5CRightarrow+%5C%5C+%5Csum_%7Bi%7D%5Csum_%7Bj%7DW_%7Bij%7D+Ax_ix_j%5ET+%3D+%28XA%5ET%29%5ETWX+%5C%5C)

又由：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bj%7DW_%7Bij%7D%3D+%5Csum_j%5Cleft%28p_ip_%7Bij%7D+-+pmask_%7Bij%7D%5Cright%29+%3D+p_i+%5Csum_j+p_%7Bij%7D+-+%5Csum_j+pmask_%7Bij%7D+%3D+p_i+-+%5Csum_%7Bj+%5Cin+C_i%7D+p_%7Bij%7D+%3D+p_i+-+p_i+%3D+0+%5C%5C)

所以 ![[公式]](https://www.zhihu.com/equation?tex=sum%28W%5ET%2C+axis+%3D+0%29+%3D+sum%28W%2C+axis+%3D+1%29+%3D+%5B0%2C+0%2C+%5Ccdots%2C+0%5D) 。这里化简得到“梯度#2”：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+2%28XA%5ET%29%5ET+%5Cleft%28+diag%5Cleft%28sum%28W%2C+axis+%3D+0%29%5Cright%29+-+W+-+W%5ET+%5Cright%29+X+%5C%5C)

注意：对比两个梯度表达形式“#1”与“#2”，后面代码实现会利用。不同的形式对应的实现在运行时间与空间上差别很大。



## **2 NCA的代码实现：四种方法**

下面介绍如何利用Python3.6来实现NCA，主要是如何快速高效地求梯度。先回顾一下两个梯度形式“#1”与“#2”：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+2A%5Csum_i+%5Cleft%28p_i%5Csum_%7Bk%5Cneq+i%7Dp_%7Bik%7D%28x_i-x_k%29%28x_i-x_k%29%5ET++-+%5Csum_%7Bj%5Cin+C_i%7D+p_%7Bij%7D%28x_i-x_j%29%28x_i-x_j%29%5ET%5Cright%29+%5C%5C)
![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%7D%7B%5Cpartial+A%7D+%3D+2%28XA%5ET%29%5ET+%5Cleft%28+diag%5Cleft%28sum%28W%2C+axis+%3D+0%29%5Cright%29+-+W+-+W%5ET+%5Cright%29+X+%5C%5C)

先来简单分析一下，在“#1”中，梯度计算是基于两层 for 循环进行的，所以可以得到最简单的实现方法，就是两层 for 循环得到梯度，在Python中利用 for 循环会很慢，所以我们还可以使用矩阵来加速操作，第一种矩阵加速是对“#1”的第二层 for 循环里面的部分使用矩阵来实现，但是会占用很多内存，第二种是利用“#2”所表达的梯度形式，实现起来很快，又不占太多内存。但是上面的实现都是利用普通的梯度下降实现的，当然还可以使用更有效的优化方法来实现，借助了scipy里面提供的一些优化方法来实现。下面详细介绍代码实现。

代码实现详见 [https://github.com/Wingspeg/NeighbourhoodComponentAnalysis-Python](https://github.com/Wingspeg/NeighbourhoodComponentAnalysis-Python)

## 代码架构
 * nca_naive.py  使用两层for循环求梯度的实现；速度很慢
 * nca_matrix.py 使用矩阵操作加速，但是仍然有一层for循环；速度稍微快了一点，但是占内存
 * nca_fast.py   使用梯度的另一种形式，全矩阵操作，没有for循环，速度很快，空间占用少
 * nca_scipy.py  使用nca_fast.py的方法 + scipy.optimize的优化包实现，分为gradient descent和coordinate descent
 * example.py    利用mnist数据集进行测试
 * usage.py      里面展示了使用方法，由于四种实现都封装为了NCA类，并且实现了类似PCA的fit, fit_transform, transform方法，使用很简单

## 方法
  * 两层for循环
  * 矩阵操作加速
  * 另一种梯度表达形式
  * scipy.optimize实现

## 2.1 实现方法一：两层 for 循环

首先，最简单地就是利用两层 for 循环遍历得到梯度：

```text
# gradients
gradients = np.zeros((self.high_dims, self.high_dims))
# for i
for i in range(self.n_samples):
    k_sum = np.zeros((self.high_dims, self.high_dims))        # first_part
    k_same_sum = np.zeros((self.high_dims, self.high_dims))   # second_part
    # for k
    for k in range(self.n_samples):
         out_prod = np.outer(X[i] - X[k], X[i] - X[k])
         k_sum += prob_mat[i][k] * out_prod
         if Y[k] == Y[i]:
             k_same_sum += prob_mat[i][k] * out_prod
    gradients += prob_row[i] * k_sum - k_same_sum
gradients = 2 * np.dot(gradients, self.A)
```

在Python中，两层 for 循环的运算速度很慢，复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%5E2%29) ，n为样本数目。上面的代码实现是最naive的方法，所以在github里面命名为nca_naive.py。



## 2.2 实现方法二：矩阵操作加速

因为 for 循环速度太慢，所以优化一部分能用矩阵操作来进行得到的计算。下面我们先看看如何尽可能地使用矩阵操作，考虑下面问题，给定样本矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X) ，如何求第 i 个样本与所有样本的差呢，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Bx_i-x_1%3Bx_i-x_2%3B%5Ccdots%3Bx_i-x_n%5D%5ET) ，当然利用 for 循环来写是可以的，但是在Python中利用广播（broadcast）操作则很容易 计算：

```text
xik = X[i] - X
```

上面的很简单，那么假设要求样本距离矩阵呢？即 ![[公式]](https://www.zhihu.com/equation?tex=D_%7Bij%7D+%3D+%7C%7Cx_i+-+x_j%7C%7C_2%5E2) 。怎么用矩阵来实现？

当然我们可以利用两层 for 循环来实现，当然也可以用一层 for 循环，然后利用上面的广播来实现：

```text
for i in range(n):
    dist_mat[i,:] = np.sum((X[i] - X) ** 2, axis = 1)
```

如何把两层 for 循环都去掉呢，都用矩阵操作？当然如果能想到下面的转换：

![[公式]](https://www.zhihu.com/equation?tex=D_%7Bij%7D+%3D+%7C%7Cx_i%7C%7C_2%5E2+%2B+%7C%7Cx_j%7C%7C_2%5E2+-+2x_i%5ETx_j+%5C%5C)

那么就有：

```text
# distance matrix
sum_row = np.sum(low_X ** 2, axis = 1)
xxt = np.dot(low_X, low_X.transpose())
dist_mat = sum_row + np.reshape(sum_row, (-1, 1)) - 2 * xxt
```

如果不转换的话可以直接用矩阵实现吗？答案是可以的，下面是解决方法：

```text
# X[None,:,:] shape = (1, n, d)
# X[:, None, :] shape = (n, 1, d)
# X[None,:,:] - X[:,None,:] shape = (n, n, d)
# sum of square from axis = 2
dist_mat = np.sum((X[None,:,:] - X[:,None,:]) ** 2, axis = 2)
```

同样地，在求梯度的过程中也可以尽可能地将 for 循环变为矩阵操作，可以提高一点速度，这里并没有减少时间复杂度，只是在实现层面上加速了一点而已。最后实现的梯度部分代码为：

```text
# gradients
part_gradients = np.zeros((self.high_dims, self.high_dims))
for i in range(self.n_samples):
    xik = X[i] - X
    prod_xik = xik[:, :, None] * xik[:, None, :]
    pij_prod_xik = pij_mat[i][:, None, None] * prod_xik
    first_part = pi_arr[i] * np.sum(pij_prod_xik, axis = 0)
    second_part = np.sum(pij_prod_xik[Y == Y[i], :, :], axis = 0)
    part_gradients += first_part - second_part
gradients = 2 * np.dot(part_gradients, self.A)
```

需要注意的是这种实现的办法会增加内存占用，比如上面介绍求距离矩阵时，会在内存产生一个(n, n, d)的矩阵，假若n 和 d都很大，那么内存开销就会产生Memory Error。因此只适用于小数据集。

## 2.3 实现方法三：利用梯度“#2”计算

上面介绍的利用梯度“#1”实现的方法一个太慢，一个太占内存。利用“#2”则会改善很多，下面直接附上代码：

```text
# gradients
weighted_pij = pij_mat_mask - pij_mat * pi_arr[:, None]      # (n_samples, n_samples)
weighted_pij_sum = weighted_pij + weighted_pij.transpose()   # (n_samples, n_samples)
np.fill_diagonal(weighted_pij_sum, -weighted_pij.sum(axis = 0))
gradients = 2 * (low_X.transpose().dot(weighted_pij_sum)).dot(X).transpose()            
```

可以看出，完全是利用矩阵操作，没有任何循环 操作，所以时间和空间都大大改善了。

## 2.4 实现方法四：利用梯度“#2”+ Scipy优化包

上面的三种方法都要自己实现梯度下降，当然理想的是用Scipy里面的优化包来实现，我们只需要把 cost 和 gradient 传给优化包就可以了，优化的过程我们就不用去干预了。借助scipy.optimize提供的minimize，fmin_cg来实现，前者是梯度下降，后者是坐标下降。最终的代码为：

```text
# fit by gradient descent
# optimizer params
optimizer_params = {'method' : 'L-BFGS-B', 
                    'fun' : self.nca_cost,
                    'args' : (X, Y),
                    'jac' : True,
                    'x0' : A.ravel(),
                    'options' : dict(maxiter = self.max_steps),
                    'tol' : self.tol}
opt = minimize(**optimizer_params)
```

下面给出坐标下降的实现：

```text
# fit by coordinate descent 
def costf(A):
    f, _ = self.nca_cost(A.reshape((self.high_dims, self.low_dims)), X, Y)
    return f
def costg(A):
    _, g = self.nca_cost(A.reshape((self.high_dims, self.low_dims)), X, Y)
    return g
self.A = fmin_cg(costf, A.ravel(), costg, maxiter = self.max_steps)
```

上面给出了几种实现，下面给出的实验结果都是基于nca_scipy.py来实现的，采用的是梯度下降。



## **3 NCA在一些数据集上的表现**

下面给出一些结果图片,前面9张是在mnist上面的结果，选取的数字类别数目分别为2至9，由于原图是784维的，所以先使用PCA降维到100维，然后再使用NCA降维到2维；后面三张是直接使用NCA在digits，breast_cancer和iris数据集降维到2维上得到的结果。

## 结果展示
  下面给出一些结果图片,前面9张是在mnist上面的结果，选取的数字类别数目分别为2至9，由于原图是784维的，所以先使用PCA降维到100维，然后再使用NCA降维到2维；后面三张是直接使用NCA在digits(numpy提供)，breast_cancer和iris数据集降维到2维上得到的结果。
  <div> 
    <table>
     <tr>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_2_digits.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_3_digits.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_4_digits.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_5_digits.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_6_digits.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_7_digits.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_8_digits.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_9_digits.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/mnist_with_10_digits.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/digits_np.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/breast_cancer.jpg"></td>
      <td><img src = "https://github.com/wingspeg/NeighbourhoodComponentAnalysis-Python/tree/main/pics/iris.jpg"></td>
     </tr>
     
    </table>
  </div>



