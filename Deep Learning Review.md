# Deep Learning复习

[TOC]

## 一.General

### 1. 不同数据集下使用微调：

   数据量少，数据相似度高，我们所做的只是修改最后几层或最终的softmax图层的输出类别**Fine-tune**。

   数据量少，数据相似度低，我们可以冻结预训练模型的初始层（比如k层），并再次训练剩余的（n-k）层。由于新数据集的相似度较低，因此根据新数据集对较高层进行重新训练具有重要意义**（冻结训练）**

   数据量大，数据相似度低，由于我们有一个大的数据集，我们的神经网络训练将会很有效。但是，由于我们的数据与用于训练我们的预训练模型的数据相比有很大不同，使用预训练模型进行的预测不会有效。因此，最好根据你的数据从头开始训练神经网络**（Training from scatch）**。

   数据量大，数据相似度高-这是**理想情况**。在这种情况下，预训练模型应该是最有效的。使用模型的最好方法是保留模型的体系结构和模型的初始权重。然后，我们可以使用在预先训练的模型中的权重来重新训练该模型。

### 2. 为什么要微调？

   用了大型数据集做训练，已经具备了**提取浅层基础特征和深层抽象特征的能力**。

### 3. 感受野是什么？
现在的一个像素对应原来的多少个像素

### 4. 说说Batch Normalization的原理及作用？

#### BN的本质：解决反向传播中的梯度爆炸和梯度消失问题

  #### 4.1 为什么需要BN操作？

一句话总结：神经网络学习过程本质就是**学习数据分布**，因为训练过程中每层神经元的权重在改变，因此每层的**数据分布**在改变，前面层训练参数的更新会导致后面层输入数据分布的改变。一旦每批训练数据的分布各不相同, 那么网络就要在每次迭代都去学习适应不同的分布, 不仅会大大降低网络的训练速度，还需要非常谨慎地去设定学习率、初始化权重、以及尽可能细致的参数更新策略
    ![img](https://lh4.googleusercontent.com/mCvTCgnOprH3sAb97HCLfibqFgQQ2aE3nAvxJB_OWUKBx755cDuqdCJoJPoedUTQu0dKAiC9UNJG8AqZUEzahHlzjVkiL709lS98k7zxBJ8nn5ht7I2ZTG8bn1XCzlJzdyu84na6#pic_center)
   在DNN中，隐藏层将输入x通过系数矩阵W相乘得到线性组合z=Wx，再通过激活函数a=sigmoid(z)，得到隐藏层的输出a（X可以为输入层输入或者上一个隐藏层的输出）。由于同一批次数据的不断传入和训练，DNN内部参数在不断改变，导致每一次隐藏层的**输入分布**不一致。也就是在训练过程中，隐层的输入分布老是变来变去，这就产生了内部协变量偏移问题(Internal Covariate Shift).针对协变量偏移问题，Google提出了Batch Normalization算法。BN通过对隐藏层线性组合输出z=Wx进行正态标准化z’=normalization(z)，再对标准化的输出z’进行尺度放缩和平移变换，使隐藏层的输出分布一致（**注意：针对z=Wx进行标准化，而不是通过激活函数a=f(z)进行标准化**)。

![img](https://lh6.googleusercontent.com/GhAr-uwanQVKz6NeuBk0j1eSFNWlDsNE1_D8-SIbNn9DkFrpqNHf9cvXW29yeO3HejO6vcYTCU6Egv9SfMNmtY-MaYAd646HK9V4SCGhTglld8rPd29ZiTkE8wtvcFE-Eit2bw2X)
    
带有BN的隐藏层图示：**将隐藏层的输出Z进行规范化，再进行尺度变换和偏移**![img](https://lh3.googleusercontent.com/HoyAKhl0wphQkA7jdfI5XO_7bn1_6cAOBHT_HR6FfBbu7cGENj5cbTcsXIq1GJQog1uU_vVbQqiaPOtXgu8lK-gvi5sZGwcuiYY-cGEM34q7w-MBPIGYh1l_pM3qfHnFbe8fG28f)
其中参数γ和β是可以通过训练得到的。而在训练中μ和σ为该batch数据z的均值和方差。在预测时，μ和σ分别使用每个batch的μ和σ的加权并平均，其中起始输入的batch的权重较低，后面输入的batch的权重较高。

#### 4.2 BN的优点？

1. BN可以把隐层神经元激活输入z=WX从变化不拘一格的正态分布拉回到了均值为0，方差为1的正态分布。使得隐藏层的输入分布一致，这解决了前面的协变量偏移问题(Internal Covariate Shift)。
2. 同时，激活函数恰恰在中间区域的梯度是最大的，由于模型使用随机梯度下降(SGD)，这使得模型的训练使不会出现梯度弥散或者梯度忽大忽小的问题，同时参数的收敛速度更快。
3.  防止过拟合

  #### 4.3为什么BN层能加快模型收敛速度？

之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致反向传播时低层神经网络的梯度消失，这是训练深层神经网络收敛越来越慢的本质原因。而BN就是通过规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入“比较敏感”的区域，这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。（比较敏感的区域就像sigmoid函数中中间虚线这部分）![img](https://lh4.googleusercontent.com/Le78GqEphM05m1Pqxx9X2OmP7XhY27LLFT8CbIcmFbFMLxPh9WUBPSLNtS12DXTLTLH42gO6Qy0S3dl6h-RXPJvXpZS_R6r_qtBxi9jtXVooGc2JK6ywPBs_N6r987dxhj2ds7Yv)

#### 4.4 BN为什么能够解决梯度消失和梯度爆炸？

**梯度消失**: [深度神经网络](https://www.zhihu.com/search?q=深度神经网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A270968652})中, 如果网络的激活函数输出很大, 而一般输出层是sigmoid激活函数,其对应的梯度就会很小(从[sigmoid函数曲线](https://www.zhihu.com/search?q=sigmoid函数曲线&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A270968652})就能看出来). 如果不加BN, **每一层输出就会不断偏移, 最后多层偏移累加, 就产生了更大的偏移**.

**梯度爆炸**: 如果某一层神经网络输入很大, 根据链式求导规则, 这一层的梯度与数据的输入成正比的, 所以就会导致梯度很大.

**总结**: 根据[链式求导法则](https://www.zhihu.com/search?q=链式求导法则&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A270968652}), 某一层神经网络的权重的梯度与两个因素有关, 一个就是后一层的梯度大小, 一个就是当前层的数据输入x. 如果后一层梯度过小(通常来说sigmoid层输出太大会导致), 当前层的梯度就会很小, 如果当前层的输入x太大(没有归一化), 就会导致梯度爆炸.

#### 4.5 为什么归一化后还要用γ和β进行放缩和平移？

总结：如果是仅仅使用归一化, 对网络某一层A的输出数据做归一化, 然后送入网络下一层B, 这样是会影响到本层网络A所学习到的特征的. 打个比方, 比如我网络中间某一层学习到特征数据本身就分布在S型激活函数的两侧, 你强制把它给我归一化处理、标准差也限制在了1, 把数据变换成分布于s函数的中间部分, 这样就相当于我这一层网络所学习到的特征分布被你搞坏了, 这可怎么办？于是引入了可学习参数γ、β, 这就是算法关键之处![img](https://lh5.googleusercontent.com/fCDQynAfs6y4XMc_rxxDFQbWAvzDR9g_fIl-xS-HRN9JOBs3TOCvYtJDbqkbuybx3A-U_xCQQKMWAR1SJ-6NBQKqmHCJcoTQ-yRWoVssz0izgZqLzgMjo5uabU-Nb2wQ3ZCzhRDJ)  我们知道在网络初始化的时候，初始的w,b一般都很小，略大于0，如果我们将 a 图的数据归一化到 c 图的原点附近，那么，网络拟合y = wx+b时，b就相对很容易就能从初始的位置找到最优值，如果在将 c 图的数据做一个小小的拉伸，转换为 d 图的数据，此时，数据之间的相对差异性变大，拟合y = wx+b这条划分线时，w相对容易从初始位置找到最优值。这样会使训练速度加快。  现在有一个问题，假如我直接对网络的每一层输入做一个符合正态分布的归一化，如果输入数据的分布本身不是呈正态分布或者不是呈该正态分布，那会这样会容易导致后边的网络学习不到输入数据的分布特征了，因为，费劲心思学习到的特征分布被这么暴力的归一化了，因此直接对每一层做归一化显示不合理。但是稍作修改，加入可训练的参数做归一化，那就是BatchNorm实现的了 。 最重要的一步，引入缩放和平移变量γ和β ,计算归一化后的值 。 如果数据都变为了正太分布，神经网络如何学习数据的分布呢？好比中间层的神经网络好不容易学习到了点东西，但是经过最初始的BatchNorm,中间层的特征变为了0-1分布，学习到的数据分布就全没了。但是加一个平移缩放就不一样了，它不会强制让你变为0-1分布，对学习到的特征没有坏影响，甚至running参数是通过你的训练集学习到的，BN通过runing参数计算出的特征不断接近数据集的数据分布。

 #### 4.6 BN的维度计算？

（N,C,H,W）分别表示 （batch size, channels, Height, Width）

实际上我们说在图像处理中，我们的batch Normalization是2d，在视频中是3d。因为**二维是将channel看成一个维度，height和width组成另一个维度。我们会默认height和width组成一个维度，而不是两个维度。**

Batch Noramlization 是想让输入满足同一个分布，以2dBN为例，就是让每张图片的相同通道的所有像素值的均值和方差相同。比如我们有两张图片（都为3通道），我们现在只说R通道，**我们希望第一张图片的R通道的均值 和 第二张图片R通道的均值相同，方差同理**。请在脑中想象分别取出每一张图片的R通道 并 叠放在一起，然后对这所有元素求均值和方差。当我们图片是3通道的时候，我们就会有3个均值和方差。

均值的计算，就是在一个批次内，将每个通道中的数字单独加起来，再除以$$N*H*W$$ 。举个例子：该批次内有3张图片，每张图片有三个通道RBG，每张图片的高、宽是H、W，那么均值就是计算3张图片R通道的像素数值的均值，再计算B通道全部像素值的均值，最后计算G通道的像素值的均值，那么对应的，均值的维度就是 C。方差的计算类似。

可训练参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%E3%80%81%5Cbeta) 的维度等于**张量的通道数**C，在上述例子中，RBG三个通道分别需要一个$\gamma$和一个$\beta$，所以$大\gamma$和$大\beta$的维度等于3。

#### 4.7 为什么BN层一般用在线性层和卷积层后面，而不是放在非线性单元后？

#### 4.8   推理阶段，BN如何使用？

BN在训练的时候可以根据Mini-Batch里的若干训练实例进行激活数值调整，但是在推理的过程中，很明显输入就只有一个实例，看不到Mini-Batch其它实例，那么这时候怎么对输入做BN呢？因为很明显一个实例是没法算实例集合求出的均值和方差的。既然没有从Mini-Batch数据里可以得到的统计量，那就想其它办法来获得这个统计量，就是均值和方差。可以用从所有训练实例中获得的统计量来代替Mini-Batch里面m个训练实例获得的均值和方差统计量，因为本来就打算用全局的统计量，只是因为计算量等太大所以才会用Mini-Batch这种简化方式的，那么在推理的时候直接用全局统计量即可。那么如何获得均值和方差的问题。很简单，因为每次做Mini-Batch训练时，都会有Mini-Batch里m个训练实例获得的均值和方差，现在要全局统计量，只要把每个Mini-Batch的均值和方差统计量记住，然后**对这些均值和方差求其对应的数学期望即可得出全局统计量**

总结：

**训练**时，均值、方差分别是**该批次**内数据相应维度的均值与方差；

**推理**时，均值、方差是**基于所有批次**的期望计算所得，公式如下：

#### 4.9 Batch Normalization的缺点？

1. batch_size较小的时候，效果差。BN的过程，使用 整个batch中样本的均值和方差来模拟整个样本的均值和方差（整个样本分布情况），比较依赖batchsize的大小，如果batchsize太小，则计算的均值、方差不足以代表整个数据分布

#### 4.10 Group Normalization

#### 4.11 Layer Normalization

#### 4.12 Instance Normalization

### 4.13 Transformer为什么用LN不用BN？

### 损失函数

#### Focal Loss

Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。



## 二.卷积神经网络CNN

### 1. 卷积神经网络核心

1. 浅层卷积提取**基础特征**，比如边缘，轮廓

2. 深层卷积提取**抽象特征**，比如脸型

### 2. Resnet

Motivation: 网络加深遇到的优化问题(网络达到一定深度后，梯度爆炸和消失带来的性能下降问题)，resnet出现前的解决方案是更好的优化方法，更好的初始化策略，BN层，ReLU激活函数

残差块使得很深的网络更容易训练，因为残差块的跳跃连接使得先训练好浅层小模型，再慢慢训练深层的大模型，使得模型不会偏离最优解

   ![image-20211130220321715](/Users/kevin/Library/Application Support/typora-user-images/image-20211130220321715.png)

   加更多的层不一定总是提高精度，比如左图中的F6模型是最大的模型，但是距离最优解f‘的距离比小模型F3要远。因此我们希望在加深神经网络层数时要像右边图中一样，能保证随着深度增加模型越来越接近最优解。

   解决方案：

   ![image-20211130220719266](/Users/kevin/Library/Application Support/typora-user-images/image-20211130220719266.png)

   通过加法，$$f(x)=x+g(x)$$,即使层数增加后的模型$$g(x)$$没能距离最优解更近，但起码没让之前的小模型(浅层)变得更差（例如上上左图所示）。

### SENet

SENet主要是学习了channel之间的相关性，筛选出了针对通道的注意力，稍微增加了一点计算量，但是效果比较好。SENet认为每个channel的feature map的作用权重不一样，于是给每个channel学习一个权重。

原始输入X经过神经网络后获得$$(H*W*C)$$的特征图，该特征图经过全局池化后得到$$(1*1*C)$$的向量，经过全链接层，ReLu，全链接层，Sigmoid之后得到每个特征通道的重要程度，再用得到的特征通道的重要程度乘回之前的$$(H*W*C)$$的特征图（**注意力机制**）

![image-20211130225230580](/Users/kevin/Library/Application Support/typora-user-images/image-20211130225230580.png)

在经过FC时，通道数进行了缩减$$(1*1*\frac{C}{r})$$,为了减小参数量，通过实验验证r为16时模型参数最少效果最佳。

将SE放在残差单元的哪个位置更优？

**残差分支之后，聚合之前**

![image-20211130230948340](/Users/kevin/Library/Application Support/typora-user-images/image-20211130230948340.png)

## 循环神经网络RNN

### RNN梯度消失真正的含义？与DNN梯度消失一样吗

## Transformer

### 输入

#### Encoder输入

##### Embedding

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211209170201006.png" alt="image-20211209170201006" style="zoom:30%;" />

例如一个词组长度为12，每一个字可以编码成$$（1*512）$$维的embedding，一个12长度的词组就组成了$$（12*512）$$维的embedding矩阵。

##### Positional Embedding

对于每个词都有一个$（1*512）$维的positional embedding，一个12长度的词组就组成了$$（12*512）$$维的positional embedding矩阵。因此整个encoder的输入就是：（下图以某个词为例子）

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211209170002772.png" alt="image-20211209170002772" style="zoom:30%;" />

#### Decoder输入

### 为什么需要位置编码？

在RNN中的是从头到尾按照顺序依次处理，天然包含了句子之间的位置信息，比如词组“我爱你”，按照我，爱，你的顺序能提取到位置信息。而Transformer中所有的数据并行处理，因此无法获取到词组中各个单词的位置信息，模型就没有办法知道每个单词在句子中的相对位置和绝对位置信息。所以引入了位置编码。

位置编码（Positional Encoding）是不需要训练的，它有一套自己的生成方式，我们只需要把位置向量加到原来的输入embedding向量中，就能让Transformer中包含句子的位置信息。

#### 公式：

​																			$$PE_{pos, 2i} = sin(\frac{pos}{10000^{2i/d_{model}}})$$

​																			$$PE_{pos, 2i+1} = cos(\frac{pos}{10000^{2i/d_{model}}})$$

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211209172937383.png" alt="image-20211209172937383" style="zoom:50%;" />

图中，以“爱”字512维的positional embedding为例，2i表示偶数位置，2i+1表示奇数位置，偶数位置用sin函数，奇数位置用cos函数

#### 为什么用sin和cos来编码位置关系？

因为三角函数有个性质：**由于三角函数的周期性，随着$pos$的增加，相同维度的值有周期性变化的特点**

​																		$$               sin(a+b) = sin(a) * cos(b) + cos(a) * sin(b)$$

​																		$$cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)$$

可以推导出，两个位置向量的点积与他们两个位置差值（即相对位置）有关，而与绝对位置无关。这个性质使得在计算注意力权重的时候(两个向量做点积)，使得相对位置对注意力发生影响，而绝对位置变化不会对注意力有任何影响，这更符合常理。

比如”我爱中华“这句话，”华“与”中“相对位置为1，华与中的相关性程度取决于相对位置值1。而如果这句话前面还有其他字符，那华和中两个字的绝对位置会变化，这个变化不会影响到中华这两个字的相关程度。

但是这里似乎有个缺陷，就是这个相对位置没有正负之分，比如"华"在"中"的后面，对于"中"字，"华"相对位置值应该是1，而"爱"在"中"的前面，相对位置仍然是1，这就没法区分到底是前面的还是后面的。

transformer的位置向量还有一种生成方式是可训练位置向量。即随机初始化一个向量，然后由模型自动训练出最可能的向量。transformer的作者指出这种可训练向量方式的效果与正玄余玄编码方式的效果差不多。在bert的代码中采用的是可训练向量方式。



##### 代码实现位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        """
        :param d_model:embedding的维度
        :param dropout: Dropout的置零比例
        :param max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        #实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)
        #初始一个位置编码矩阵，pe维度为max_len*d_model
        pe = torch.zeros(max_len,d_model)
        #初始化一个绝对位置矩阵，词汇的位置就是用它的索引表示，position维度为max_len*1
        position = torch.arange(0,max_len).unsqueeze(1)#由[0,1,2...max_len] -> [[0],[1]...[max_len]]

        #定义一个变换矩阵使得position的[max_len,1]*变换矩阵得到pe[max_len,d_model]->变换矩阵格式[1,d_model]
        #除以这个是为了加快收敛速度
        #div_term格式是[0,1,2...d_model/2],分成了两个部分，步长为2
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))

        #将前面定义好的矩阵进行奇数偶数赋值
        pe[:,0::2] = torch.sin(position*div_term) # 从0开始步长为2（偶数）
        pe[:,1::2] = torch.cos(position*div_term) # 从1开始步长为2（奇数）

        #此时pe[max_len,d_model]
        #embedding三维(可以是[batch_size,vocab,d_model])#vocab就是max_len
        #将pe升起一个维度扩充成三维张量
        pe = pe.unsqueeze(0)

        #位置编码矩阵注册成模型的buffer，它不是模型中的参数，不会跟随优化器进行优化
        #注册成buffer后我们就可以在模型的保存和加载时，将这个位置编码器和模型参数加载进来
        self.register_buffer('pe',pe)


    def forward(self,x):
        """
        :param x:x代表文本序列的词嵌入
        pe编码过长将第二个维度也就是max_len的维度缩小成句子的长度
        """
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)


x = torch.LongTensor([[1,2,3,4],[5,6,7,8]])#[2,4]
emb = Embeddings(d_model,vocab)#[2,4,512]
embr = emb(x)
pe = PositionalEncoding(d_model,dropout=0.2,max_len=50)
pe_result = pe(embr)
#
print(pe_result)
print(pe_result.shape)#[2,4,512]
```





