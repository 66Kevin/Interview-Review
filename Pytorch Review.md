[TOC]

# Pytorch Review

## 1. Pytorch 数据

由于Python的缓存协议,只要PyTorch的数据是在cpu上,不在GPU上,那么torch.Tensor类型的数据和numpy.ndarray的数据是共享内存的,相互之间的改变相互影响.

### 1.1 numpy转tensor

```python
>>> import numpy as np
>>> import torch
>>> points_np = np.ones(12).reshape(3,4)
>>> points_np
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
>>> points = torch.from_numpy(points_np)  						# 转换成cpu的tensor数据
>>> points_cuda = torch.from_numpy(points_np).cuda() 	# 转换成gpu的tensor数据
>>> ##########################################################
>>> points_np
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
>>> points
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype=torch.float64)
>>> points_cuda
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], device='cuda:0', dtype=torch.float64)
>>> id(points_np)
1999751313328  # numpy与cpu的tensor共享内存
>>> id(points)
1999751939640	 # numpy与cpu的tensor共享内存
>>> id(points_cuda)
1999804358072
```

### 1.2 CUDA tensor转numpy

```python
>>> import torch
>>> points_cuda = torch.ones(3, 4).cuda()
>>> points_cpu = points_cuda.cpu()
>>> points_np = points_cuda.cpu().numpy() # 不能把CUDA tensor直接转为numpy. 应先使用Tensor.cpu()把tensor拷贝到memory.
>>> id(points_np)
1990030518992
>>> id(points_cpu)
1989698386344
>>> id(points_cuda)
1990030519736
```

如果 tensor 需要求导的话，还需要加一步 detach，再转成 Numpy 。

```python
x  = torch.rand([3,3], device='cuda')
x_ = x.cpu().numpy()

y  = torch.rand([3,3], requires_grad=True, device='cuda').
y_ = y.cpu().detach().numpy()
# y_ = y.detach().cpu().numpy() 也可以
```

### 1.3 CPU tensor转numpy

```python
>>> import torch
>>> points_cpu = torch.ones(3, 4)
>>> points_np = points_cpu.numpy()
>>> points_cpu
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
>>> points_np
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)
>>> id(points_cpu)==id(points_np)
False
>>> type(points_cpu)
<class 'torch.Tensor'>
>>> id(points_np)
1291906427600
>>> type(points_np)
<class 'numpy.ndarray'>
```



### 1.4 numpy 转 tensors效率高

tensor和numpy是共享相同内存的，两者之间转换很快

Code 1:

```
import torch
import numpy as np

a = [np.random.randint(0, 10, size=(7, 7, 3)) for _ in range(100000)]
b = torch.tensor(np.array(a))
```

Code 2:

```
import torch
import numpy as np

a = [np.random.randint(0, 10, size=(7, 7, 3)) for _ in range(100000)]
b = torch.tensor(a)
```

```
The code 1 takes less than 1 second to execute (used time):

real    0m0,915s
user    0m0,808s
sys     0m0,330s

Whereas the code 2 takes 5 seconds:

real    0m6,057s
user    0m5,979s
sys     0m0,308s
```



### 1.5 PyTorch 把tensor转换成int

直接在tensor变量的后面加.item()，就能把tensor类型转换成int类型。

## Pytorch的动态计算图

计算图是用来描述运算的有向无环图；

计算图有两个主要元素：结点（Node）和边（Edge）；

结点表示数据，如向量、矩阵、张量，边表示运算，如加减乘除卷积等；

用计算图表示：$$y = ( x + w ) ∗ ( w + 1 )$$
令$$a=x+w , b=w+1, y=a∗b$$，那么得到的计算图如下所示：

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211216153652682.png" alt="image-20211216153652682" style="zoom:43%;" />

数学求导：

$$ y=(x+w)∗(w+1)$$
$$a = x + w; b = w + 1$$
$$y = a ∗ b$$

$$\frac{\partial y}{\partial w} = \frac{\partial y}{\partial a} \frac{\partial a}{\partial w} + \frac{\partial y}{\partial b} \frac{\partial b}{\partial w}$$

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211216154411949.png" alt="image-20211216154411949" style="zoom:44%;" />

通过分析可以知道，y对w求导就是在计算图中找到所有y到w的路径，把路径上的导数进行求和。



其中，叶子节点是x和w，叶子节点是整个计算图的根基。

什么是叶子节点张量呢？叶子节点张量需要满足两个条件。

1，叶子节点张量是由用户直接创建的张量，而非由某个Function通过计算得到的张量。

2，叶子节点张量的 requires_grad属性必须为True.

例如前面求导的计算图，在前向传导中的a、b和y都要依据创建的叶子节点x和w进行计算的。同样，在反向传播过程中，所有梯度的计算都要依赖叶子节点。设置叶子节点主要是为了节省内存，在梯度反向传播结束之后，非叶子节点的梯度都会被释放掉。


所有依赖于叶子节点张量的张量, 其requires_grad 属性必定是True的，但其梯度值只在计算过程中被用到，不会最终存储到grad属性中。如果需要保留中间计算结果的梯度到grad属性中，可以使用 retain_grad方法。

```python
import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
print("loss.grad:", loss.grad)
print("y1.grad:", y1.grad) 
print("y2.grad:", y2.grad)
print(x.grad)

# loss.grad: None  # loss是非叶子张量，需用.retain_grad()方法保留导数，否则导数将会在反向传播完成之后被释放掉
# y1.grad: None	   # y1是非叶子张量，需用.retain_grad()方法保留导数，否则导数将会在反向传播完成之后被释放掉
# y2.grad: None	   # y2是非叶子张量，需用.retain_grad()方法保留导数，否则导数将会在反向传播完成之后被释放掉
# tensor(4.)
```

```python
print(x.is_leaf)
print(y1.is_leaf)
print(y2.is_leaf)
print(loss.is_leaf)

# True
# False
# False
# False
```



<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211216011059363.png" alt="image-20211216011059363" style="zoom:40%;" />

Pytorch中的计算图是动态图。这里的动态主要有两重含义。

第一层含义是：计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。

第二层含义是：计算图在反向传播后立即销毁。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，或者利用torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。

**计算图在反向传播后立即销毁:**

```python
import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y,2))

#计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
loss.backward()  #loss.backward(retain_graph = True) 

#loss.backward() #如果再次执行反向传播将报错
```

## requires_grad，grad_fn，grad的含义及使用

- **requires_grad**: 如果需要为张量计算梯度，则为True，否则为False。我们使用pytorch创建tensor时，可以指定requires_grad为True（默认为False），

- **grad_fn**： grad_fn用来记录变量是怎么来的，方便计算梯度，例如y = x*3, grad_fn记录了y由x计算的过程。

- **grad**：当执行完了backward()之后，通过x.grad查看x的梯度值。

创建一个Tensor并设置requires_grad=True，requires_grad=True说明该变量需要计算梯度。

```python
>>x = torch.ones(2, 2, requires_grad=True)
 
tensor([[1., 1.],
 
        [1., 1.]], requires_grad=True)
 
>>print(x.grad_fn)  # None
```

```python
>>y = x + 2
 
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward>)
 
>>print(y.grad_fn)  # <AddBackward object at 0x1100477b8>
```

**由于x是直接创建的，所以它没有grad_fn，而y是通过一个add操作创建的，所以y有grad_fn**

像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None

```python
>>z = y * y * 3
 
>>out = z.mean()
 
>>print(z, out)
```

当我们对out使用backward()方法后，就可以查看x的梯度值

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
y.retain_grad() # y是非叶子张量，需用.retain_grad()方法保留导数，否则导数将会在反向传播完成之后被释放掉, 就无法打印出y.grad
    
out = z.mean()
out.retain_grad() # y是非叶子张量，需用.retain_grad()方法保留导数，否则导数将会在反向传播完成之后被释放掉, 就无法打印出out.grad

out.backward()

print(x.grad)
print(y.grad)
print(out.grad)
```

输出：

```shell
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
        
tensor(1.)
```

数学求导过程：

$$x = \left[
 \begin{matrix}
   1 & 1  \\
   1 & 1  \\
  \end{matrix}
  \right]$$

$$y = \left[
 \begin{matrix}
   3 & 3  \\
   3 & 3  \\
  \end{matrix}
  \right]$$

$$z = \left[
 \begin{matrix}
   27 & 27  \\
   27 & 27  \\
  \end{matrix}
  \right]$$

$$out = \frac{1}{4} *z = \frac{1}{4}*y^2*3 = \frac{3}{2}y^2$$

$$\partial out = \frac{3}{2} y$$

$$\frac{\partial out}{\partial x} = \left[
 									\begin{matrix}
   											\frac{3}{2} y_{11} & \frac{3}{2} y_{12}  \\
  									 		\frac{3}{2} y_{21} & \frac{3}{2} y_{22}  \\
  											\end{matrix}
  									\right] = \left[
 									\begin{matrix}
   											\frac{9}{2}& \frac{9}{2}  \\
  									 		\frac{9}{2} & \frac{9}{2}  \\
  											\end{matrix}
  									\right]$$

同理可得出，

$$\frac{\partial out}{\partial y}$$

$$\frac{\partial out}{\partial out}$$

**注意：grad在反向传播过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。**



## PyTorch中的自动求导系统如何跟踪梯度？

Pytorch中的Tensor的属性```.requires_grad```如果设置为True,它将开始追踪在其上的所有操作（这样就可以利用链式法则进行梯度传播）。完成计算后，可以调用.backward()来完成所有梯度计算。此Tensor的梯度将累积到.grad属性中

```python
# 这里需要注意，Only Tensors of floating point and complex dtype can require gradients
x = torch.tensor([1.,2.],requires_grad=True)
print(x)  #tensor([1., 2.], requires_grad=True)
print(x.grad_fn) #None
y = x + 1
print(y)  #tensor([2., 3.], grad_fn=<AddBackward0>)
print(y.grad_fn) #<AddBackward0 object at 0x0000027D57859D48>
z = y ** 2
print(z.grad_fn) #<PowBackward0 object at 0x0000027D57859D48>
```

这里因为x是直接创建的，所以```grad_fn```为None, 而y是通过一个加法操作创建的，所以它有一个的```grad_fn```，z是通过幂指数操作创建的，所以它有一个的grad_fn。

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
#方法一：
# z = z.sum() #这样得到的就是一个标量
# z.backward()
# print(x.grad) #tensor([2.0000, 0.2000, 0.0200, 0.0020])

#方法二：
#现在 z 不是一个标量，所以在调用backward时需要传入一个和z同形的权重向量进行加权求和得到一个标量
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad) #tensor([2.0000, 0.2000, 0.0200, 0.0020])

```

```python
#如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我么可以对tensor.data进行操作。
x = torch.ones(1,requires_grad=True)
print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外
y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)

# 以下为输出
tensor([1.])
False
tensor([100.], requires_grad=True)
tensor([2.])
```

## PyTorch中的 tensor 及使用

引用自：[(极市平台)浅谈PyTorch 中的 tensor 及使用](https://mp.weixin.qq.com/s/6Q6LrRwGmGZ7Qs72hVNE7A)

### tensor.requires_grad

当我们创建一个张量 (tensor) 的时候，如果没有特殊指定的话，那么这个张量是默认是**不需要**求导的。我们可以通过 `tensor.requires_grad` 来检查一个张量是否需要求导。

在张量间的计算过程中，如果在所有输入中，有一个输入需要求导，那么输出一定会需要求导；相反，只有当所有输入都不需要求导的时候，输出才会不需要。

举一个比较简单的例子，比如我们在训练一个网络的时候，我们从 `DataLoader` 中读取出来的一个 `mini-batch`的数据，这些输入默认是不需要求导的，其次，网络的输出我们没有特意指明需要求导吧，Ground Truth 我们也没有特意设置需要求导吧。这么一想，哇，那我之前的那些 loss 咋还能自动求导呢？其实原因就是上边那条规则，虽然输入的训练数据是默认不求导的，但是，**我们的 model 中的所有参数，它默认是求导的**，这么一来，其中只要有一个需要求导，那么输出的网络结果必定也会需要求的。来看个实例：

```python
input = torch.randn(8, 3, 50, 100)
print(input.requires_grad)
# False 默认不需要求导的

net = nn.Sequential(nn.Conv2d(3, 16, 3, 1),
                    nn.Conv2d(16, 32, 3, 1))
for param in net.named_parameters():
    print(param[0], param[1].requires_grad)
# 0.weight True
# 0.bias True
# 1.weight True
# 1.bias True

output = net(input)
print(output.requires_grad)
# True
```

请注意前边只是举个例子来说明。在写代码的过程中，**不要**把网络的输入和 Ground Truth 的 `requires_grad` 设置为 True。虽然这样设置不会影响反向传播，但是需要额外计算网络的输入和 Ground Truth 的导数，增大了计算量和内存占用不说，这些计算出来的导数结果也没啥用。因为我们只需要神经网络中的参数的导数，用来更新网络，其余的导数都不需要。

我们试试把网络参数的 `requires_grad` 设置为 False 会怎么样，同样的网络：

```python
input = torch.randn(8, 3, 50, 100)
print(input.requires_grad)
# False

net = nn.Sequential(nn.Conv2d(3, 16, 3, 1),
                    nn.Conv2d(16, 32, 3, 1))
for param in net.named_parameters():
    param[1].requires_grad = False
    print(param[0], param[1].requires_grad)
# 0.weight False
# 0.bias False
# 1.weight False
# 1.bias False

output = net(input)
print(output.requires_grad)
# False
```

这样有什么用处？用处大了。我们可以通过这种方法，在训练的过程中冻结部分网络，让这些层的参数不再更新，这在**迁移学习**中很有用处。我们来看一个[Tutorial—FINETUNING TORCHVISION MODELS](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html%23initialize-and-reshape-the-networks )给的例子：

```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# 用一个新的 fc 层来取代之前的全连接层
# 因为新构建的 fc 层的参数默认 requires_grad=True
model.fc = nn.Linear(512, 100)

# 只更新 fc 层的参数
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# 通过这样，我们就冻结了 resnet 前边的所有层，
# 在训练过程中只更新最后的 fc 层中的参数。
```

### torch.no_grad()

当我们在做 evaluating 的时候（不需要计算导数），我们可以将推断（inference）的代码包裹在 `with torch.no_grad():` 之中，以达到**暂时**不追踪网络参数中的导数的目的，总之是为了减少可能存在的计算和内存消耗。看 https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients 给出的例子：

```python
x = torch.randn(3, requires_grad = True)
print(x.requires_grad)
# True
print((x ** 2).requires_grad)
# True

with torch.no_grad():
    print((x ** 2).requires_grad)
    # False

print((x ** 2).requires_grad)
# True
```

### 反向传播及网络的更新

这部分我们比较简单地讲一讲，有了网络输出之后，我们怎么根据这个结果来更新我们的网络参数呢。我们以一个非常简单的自定义网络来讲解这个问题，这个网络包含2个卷积层，1个全连接层，输出的结果是20维的，类似分类问题中我们一共有20个类别，网络如下：

```python
class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=False)
        self.linear = nn.Linear(32*10*10, 20, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x.view(x.size(0), -1))
        return x
```

接下来我们用这个网络，来研究一下整个网络更新的流程：

```python
# 创建一个很简单的网络：两个卷积层，一个全连接层
model = Simple()
# 为了方便观察数据变化，把所有网络参数都初始化为 0.1
for m in model.parameters():
    m.data.fill_(0.1)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

model.train()
# 模拟输入8个 sample，每个的大小是 10x10，
# 值都初始化为1，让每次输出结果都固定，方便观察
images = torch.ones(8, 3, 10, 10)
targets = torch.ones(8, dtype=torch.long)

output = model(images)
print(output.shape)
# torch.Size([8, 20])

loss = criterion(output, targets)

print(model.conv1.weight.grad)
# None
loss.backward()
print(model.conv1.weight.grad[0][0][0])
# tensor([-0.0782, -0.0842, -0.0782])
# 通过一次反向传播，计算出网络参数的导数，
# 因为篇幅原因，我们只观察一小部分结果

print(model.conv1.weight[0][0][0])
# tensor([0.1000, 0.1000, 0.1000], grad_fn=<SelectBackward>)
# 我们知道网络参数的值一开始都初始化为 0.1 的

optimizer.step()
print(model.conv1.weight[0][0][0])
# tensor([0.1782, 0.1842, 0.1782], grad_fn=<SelectBackward>)
# 回想刚才我们设置 learning rate 为 1，这样，
# 更新后的结果，正好是 (原始权重 - 求导结果) ！

optimizer.zero_grad()
print(model.conv1.weight.grad[0][0][0])
# tensor([0., 0., 0.])
# 每次更新完权重之后，我们记得要把导数清零啊，
# 不然下次会得到一个和上次计算一起累加的结果。
# 当然，zero_grad() 的位置，可以放到前边去，
# 只要保证在计算导数前，参数的导数是清零的就好。
```

这里，我们多提一句，我们把整个网络参数的值都传到 `optimizer` 里面了，这种情况下我们调用 `model.zero_grad()`，效果是和 `optimizer.zero_grad()` 一样的。这个知道就好，建议大家坚持用 `optimizer.zero_grad()`。具体原因具体请移步：[每一轮batch中需要必须做的事](#within_batch)](#within_batch) 我们现在来看一下如果没有调用 zero_grad()，会怎么样吧：

```python
# ...
# 代码和之前一样
model.train()

# 第一轮
images = torch.ones(8, 3, 10, 10)
targets = torch.ones(8, dtype=torch.long)

output = model(images)
loss = criterion(output, targets)
loss.backward()
print(model.conv1.weight.grad[0][0][0])
# tensor([-0.0782, -0.0842, -0.0782])

# 第二轮
output = model(images)
loss = criterion(output, targets)
loss.backward()
print(model.conv1.weight.grad[0][0][0])
# tensor([-0.1564, -0.1684, -0.1564])
```

我们可以看到，第二次的结果正好是第一次的2倍。第一次结束之后，因为我们没有更新网络权重，所以第二次反向传播的求导结果和第一次结果一样，加上上次我们没有将 loss 清零，所以结果正好是2倍。具体请移步：[每一轮batch中需要必须做的事](#within_batch)

###  tensor.detach()

当我们训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者只训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播。

假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B. 那么可以这样做：

```input_B = output_A.detach()```

**它可以使两个计算图的梯度传递断开**，从而实现我们所需的功能。

返回一个新的tensor，新的tensor和原来的tensor**共享数据内存**，**但不涉及梯度计算**，即requires_grad=False。修改其中一个tensor的值，另一个也会改变，因为是共享同一块内存,但如果对其中一个tensor执行某些内置操作，则会报错，例如resize_、resize_as_、set_、transpose_。

```python
>>> import torch
>>> a = torch.rand((3, 4), requires_grad=True)
>>> b = a.detach()
>>> id(a), id(b)  # a和b不是同一个对象了
(140191157657504, 140191161442944)
>>> a.data_ptr(), b.data_ptr()  # 但指向同一块内存地址
(94724518609856, 94724518609856)
>>> a.requires_grad, b.requires_grad  # b的requires_grad为False
(True, False)
>>> b[0][0] = 1
>>> a[0][0]  # 修改b的值，a的值也会改变
tensor(1., grad_fn=<SelectBackward>)
>>> b.resize_((4, 3))  # 报错
RuntimeError: set_sizes_contiguous is not allowed on a Tensor created from .data or .detach().
```

#### tensor.detach()与tensor.data()

在 0.4.0 版本以前，`.data` 是用来取 `Variable` 中的 `tensor` 的，但是之后 `Variable` 被取消，`.data` 却留了下来。现在我们调用 `tensor.data`，可以得到 tensor的数据 + `requires_grad=False` 的版本，而且二者共享储存空间，也就是如果修改其中一个，另一个也会变。因为 PyTorch 的自动求导系统不会追踪 `tensor.data` 的变化，所以使用它的话可能会导致求导结果出错。官方建议使用 `tensor.detach()` 来替代它，二者作用相似，但是 detach 会被自动求导系统追踪，使用起来很安全。多说无益，我们来看个例子吧：

```python
a = torch.tensor([7., 0, 0], requires_grad=True)
b = a + 2
print(b)
# tensor([9., 2., 2.], grad_fn=<AddBackward0>)

loss = torch.mean(b * b)

b_ = b.detach()
b_.zero_()
print(b)
# tensor([0., 0., 0.], grad_fn=<AddBackward0>)
# 储存空间共享，修改 b_ , b 的值也变了

loss.backward()
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

这个例子中，`b` 是用来计算 loss 的一个变量，我们在计算完 loss 之后，进行反向传播之前，修改 `b` 的值。这么做会导致相关的导数的计算结果错误，因为我们在计算导数的过程中还会用到 `b` 的值，但是它已经变了（和正向传播过程中的值不一样了）。在这种情况下，PyTorch 选择报错来提醒我们。但是，如果我们使用 `tensor.data` 的时候，结果是这样的：

```python
a = torch.tensor([7., 0, 0], requires_grad=True)
b = a + 2
print(b)
# tensor([9., 2., 2.], grad_fn=<AddBackward0>)

loss = torch.mean(b * b)

b_ = b.data
b_.zero_()
print(b)
# tensor([0., 0., 0.], grad_fn=<AddBackward0>)

loss.backward()

print(a.grad)
# tensor([0., 0., 0.])

# 其实正确的结果应该是：
# tensor([6.0000, 1.3333, 1.3333])
```

这个导数计算的结果明显是错的，但没有任何提醒，之后再 Debug 会非常痛苦。所以，建议大家都用 `tensor.detach()` 啊。



## tensor.item()和tensor.tolist()

我们在提取 loss 的纯数值的时候，常常会用到 `loss.item()`，其返回值是一个 Python 数值 (python number)。不像从 tensor 转到 numpy (需要考虑 tensor 是在 cpu，还是 gpu，需不需要求导)，无论什么情况，都直接使用 `item()` 就完事了。如果需要从 gpu 转到 cpu 的话，PyTorch 会自动帮你处理。

但注意 `item()` 只适用于 tensor 只包含一个元素的时候。因为大多数情况下我们的 loss 就只有一个元素，所以就经常会用到 `loss.item()`。如果想把含多个元素的 tensor 转换成 Python list 的话，要使用 `[tensor.tolist](<https://www.zhihu.com/search?q=tensor.tolist&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType":"article","sourceId":67184419}>)()`。

```python
x  = torch.randn(1, requires_grad=True, device='cuda')
print(x)
# tensor([-0.4717], device='cuda:0', requires_grad=True)

y = x.item()
print(y, type(y))
# -0.4717346727848053 <class 'float'>

x = torch.randn([2, 2])
y = x.tolist()
print(y)
# [[-1.3069953918457031, -0.2710231840610504], [-1.26217520236969, 0.5559719800949097]]
```





## <span id="within_batch">每一轮batch中梯度需要必须做的事</span>

### optimizer.zero_grad()

将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

在学习pytorch的时候注意到，对于每个batch大都执行了这样的操作：

```python
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
 
for epoch in range(1, epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        output= model(inputs) # 前向推理
        loss = criterion(output, labels) # 计算loss
        # 处理梯度
        optimizer.zero_grad() # 梯度归零
        loss.backward() 	## 反向传播求解梯度
        optimizer.step()  ## 梯度下降来更新权重参数
```
源码：

**param_groups**：Optimizer类在实例化时会在构造函数中创建一个param_groups列表，列表中有num_groups个长度为6的param_group字典（num_groups取决于你定义optimizer时传入了几组参数），每个param_group包含了 ['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'] 这6组键值对。

​	**param_groups**：[param_group(dict), param_group(dict), param_group(dict), param_group(dict) ... ... param_group(dict)]

​	其中每个param_group包含了 ['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'] 这6组键值对。

**param_group['params']**：由传入的模型参数组成的列表，即实例化Optimizer类时传入该group的参数，如果参数没有分组，则为整个模型的参数model.parameters()，每个参数是一个torch.nn.parameter.Parameter对象。

```python
 def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
```

optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。

因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。

**注意：一定条件下，batchsize越大训练效果越好，如果不是每一个batch就清除掉原有的梯度，而是比如说两个batch再清除掉梯度，这是一种变相提高batch_size的方法，如果8次迭代中未清理梯度，则batchsize '变相' 扩大了8倍，是我们这种乞丐实验室解决显存受限的一个不错的trick，使用时需要注意，学习率也要适当放大。**对于计算机硬件不行，但是batch_size可能需要设高的领域比较适合，比如目标检测模型的训练

### loss.backward()

PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。

具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算(利用链式法则)，如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。

更具体地说，损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。

如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。

### optimizer.step()：

step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以**在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。**

**注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。**

引用自[理解optimizer.zero_grad(), loss.backward(), optimizer.step()的作用及原理][https://blog.csdn.net/PanYHHH/article/details/107361827]



自动微分与优化器求最小值：

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c的最小值

x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)


def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()


print("y=",f(x).data,";","x=",x.data)
```

## view函数的作用

重构张量的维度，相当于numpy中resize（）的功能，但是用法可能不太一样。

```python
>>> import torch
>>> tt1=torch.tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
>>> result=tt1.view(3,2)
>>> result
tensor([[-0.3623, -0.6115],
        [ 0.7283,  0.4699],
        [ 2.3261,  0.1599]])
```
```python
>>> import torch
>>> tt2=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt2.view(-1) 一行
>>> result
tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
```

```python
>>> import torch
>>> tt3=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt3.view(2,-1)
>>> result
tensor([[-0.3623, -0.6115,  0.7283],
        [ 0.4699,  2.3261,  0.1599]])
```

由上面的案例可以看到，如果是torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。



## torch.cat(data,dim)函数

```python
>>> import torch
>>> A=torch.ones(2,3) #2x3的张量（矩阵）                                     
>>> A
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
>>> B=2*torch.ones(4,3)#4x3的张量（矩阵）                                    
>>> B
tensor([[ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.],
        [ 2.,  2.,  2.]])
>>> C=torch.cat((A,B),0) #按维数0（行）拼接
>>> C
tensor([[ 1.,  1.,  1.],
         [ 1.,  1.,  1.],
         [ 2.,  2.,  2.],
         [ 2.,  2.,  2.],
         [ 2.,  2.,  2.],
         [ 2.,  2.,  2.]])
>>> C.size()
torch.Size([6, 3])
>>> D=2*torch.ones(2,4) #2x4的张量（矩阵）
>>> C=torch.cat((A,D),1)#按维数1（列）拼接
>>> C
tensor([[ 1.,  1.,  1.,  2.,  2.,  2.,  2.],
        [ 1.,  1.,  1.,  2.,  2.,  2.,  2.]])
>>> C.size()
torch.Size([2, 7])
```

1. ```C=torch.cat((A,B),0)```就表示按维数0（行）拼接A和B，也就是竖着拼接，A上B下。此时需要注意：列数必须一致
2. ```C=torch.cat((A,B),1)```就表示按维数1（列）拼接A和B，也就是横着拼接，A左B右。此时需要注意：行数必须一致

## 数据读取```__getitem__```的顺序问题：

```def __getitem__(self,idx):```
idx的范围是从0到len-1（__len__的返回值）

但是如果采用了dataloader进行迭代，num_workers大于一的话，因为是多线程，所以运行速度不一样，这个时候如果在__getitem__函，数里输出idx的话，就是乱序的。但是实际上当线程数设置为1还是顺序的。

## 保存和加载训练模型

```python
state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,'stage': stage}
torch.save(state, "/path/...")
```

1. ```torch.save```是把一个dict保存下来，这个dict可以自定义，如上面代码中的state所示，state中```"state_dict"```是保存的模型的参数```model.state_dict()```, 其他的信息比如epoch，loss，lr都可以自定义。
2. 仅保存学习到的参数：```torch.save(model.state_dict(), "/path/...")```

```python
model = models.ECGNet() # 加载自定义的模型骨架
ecgnet = torch.load("xxx.pth") # 加载的是训练好的模型
model.load_state_dict(ecgnet['state_dict'])
# load_state_dict把torch.load加载的参数加载到模型中
```

其中，

1. ```ecgnet = torch.load("xxx.pth")```是把之前自定义的dict全部加载出来，如上图代码中的state字典，state字典中的```"state_dict"```才是真正的模型参数。接下来的```model.load_state_dict(ecgnet['state_dict'])```才是真正的把模型参数加载到model模型骨架中。

2. ```model.load_state_dict()```只能加载模型的参数，不能加载epoch，loss其他的信息，因此如果之前没有自定义保存的state字典，则读取时可以直接```model.load_state_dict(ecgnet = torch.load("xxx.pth"))```

3. 仅加载模型的某一层参数：```conv1_weight_state = torch.load("xxx.pth")['conv1.weight']```

4. ```model.load_state_dict(state_dict, strict=False)```, 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合；如果新构建的模型在层数上进行了部分微调，则上述代码就会报错：说key对应不上。此时，如果我们采用strict=False 就能够完美的解决这个问题。也即，与训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化。

   举例：假设 trained.pth 是一个训练好的网络的模型参数存储，model = Net()是我们刚刚生成的一个新模型，我们希望将trained.pth中的参数加载到新模型中，但是model中多了一些trained.pth中不存在的参数，会报错，说key对应不上，因为model你强人所难，trained.pth没有你的那些个零碎玩意，你非要向我索取，我上哪给你弄去。但是model不干，说既然你不能完全满足我的需要，那么你有什么我就拿什么吧，怎么办呢？strict=False 即可

   

## 加载预训练模型

1. 定义网络结构

   ```python
   # DenseNet这个类就是网络denesnet的结构的定义，这里参考了pytorch里面models的源码
   class DenseNet(nn.Module）：
       ... ...(此处网络结构的定义省略
   class fw_DenseNet(nn.Module）：
       ... ...(这个是我修改后网络结构）
   ```

2. 获取网络参数

   ```python
   '''预训练的模型'''
   net = DenseNet()
   net.load_state_dict(torch.load('/home/wei.fan/.torch/models/densenet161-17b70270.pth'))
   net_dict = net.state_dict()  #获取预训练模型的参数
   ''' 自定义的网络模型'''
   net1 = fw_DenseNet()
   net1_dict = net1.state_dict() #获取参数，但其实没有参数（因为没有训练）
   ```

3. 使用预训练模型参数更新自定义模型的参数

   ```python
   net1_dict = {k: v for k, v in net_dict.items() if k in net1_dict} #把两个模型中名称不同的层去掉
   net1_dict.update(net1_dict) #使用预训练模型更新新模型的参数
   net1.load_state_dict(net1_dict) #更新模型
   ```

   1. ```d1.update(d2)```: d1.update(d2)的作用是,将字典d2的内容合并到d1中,其中d2中的键值对但d1中没有的键值对会增加到d1中去,两者都有的键值对更新为d2的键值对.
   
   2. ```python
      dict = {'Name': 'Zara', 'Age': 7}
      for k, v in dict.items():
          print(k)
          print(v)
      ```

		output:  Name
						Zara
						Age
						7
		
	3. ```{k: v for k, v in net_dict.items() if k in net1_dict}```相当于先把net_dict中的key和value取出，如果key在net1_dict中出现过，则保留k和v。



## 遍历网络

### models()

先定义一个简单的网络：

```python
lass Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(64,64,3)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.features = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(64,128,3)),
            ('conv4', nn.Conv2d(128,128,3)),
            ('relu1', nn.ReLU())
        ]))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.features(x)

        return x
```

```python
    m = Model()
    for idx,m in enumerate(m.modules()): # modules()返回一个包含当前模型所有模块的迭代器，这个是递归的返回网络中的所有Module
        print(idx,"-",m)
```
结果：

```python
0 - Model(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (features): Sequential(
    (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (relu1): ReLU()
  )
)
1 - Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
2 - Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
3 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
4 - Sequential(
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
  (relu1): ReLU()
)
5 - Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
6 - Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
7 - ReLU()
```

1. `0-Model` 整个网络模块
2. `1-2-3-4` 为网络的4个子模块，注意`4 - Sequential`仍然包含有子模块
3. `5-6-7`为模块`4 - Sequential`的子模块

可以看出`modules()`是递归的返回网络的各个module，**从最顶层直到最后的叶子module**

### named_models()

`named_modules()`的功能和`modules()`的功能类似，不同的是它返回内容有两部分:module的名称以及module。

```bash
for name, layer in model.named_modules():
    if 'conv' in name:
        对layer进行处理
```

### children()

和`modules()`不同，`children()`只返回当前模块的子模块，不会递归子模块。

```python
    for idx,m in enumerate(m.children()):
        print(idx,"-",m)
```

其输出为：

```python
0 - Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
1 - Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
2 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
3 - Sequential(
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
  (relu1): ReLU()
)
```

子模块`3-Sequential`仍然有子模块，`children()`没有递归的返回。

###  named_children()

`named_children()`和`children()`的功能类似，不同的是其返回两部分内容：模块的名称以及模块本身。

## 网络的参数

### parameters()

方法`parameters()`返回一个包含模型所有参数的迭代器。一般用来当作optimizer的参数。

```python
class Net(torch.nn.Module):  # 继承torch的module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__功能
        # 定义每一层用什么样的样式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        # 激励函数（隐藏层的线性值）
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.predict(x)  # 输出值
        return x

net = Net(2, 5, 3)
paras = list(net.parameters())
for num,para in enumerate(paras):
    print('number:',num)
    print(para)
```
输出：

```python
number: 0
Parameter containing:
tensor([[ 0.1853, -0.6271],
        [ 0.5409, -0.0939],
        [ 0.6861,  0.4832],
        [ 0.3277, -0.4269],
        [ 0.4698, -0.1261]], requires_grad=True)
number: 1
Parameter containing:
tensor([-0.4354,  0.2024,  0.4123,  0.2627, -0.4142], requires_grad=True)
number: 2
Parameter containing:
tensor([[ 0.3494, -0.2752, -0.3877, -0.0560, -0.2762],
        [-0.3908,  0.1223,  0.0575, -0.0628, -0.3021],
        [ 0.4341,  0.2195, -0.1508, -0.1753,  0.1300],
        [-0.1991, -0.1677, -0.3903,  0.0275,  0.4424],
        [ 0.2589, -0.1709, -0.0964,  0.3483,  0.2197]], requires_grad=True)
number: 3
Parameter containing:
tensor([-0.1015, -0.0105, -0.1819, -0.2626, -0.2945], requires_grad=True)
number: 4
Parameter containing:
tensor([[-0.1568, -0.3360, -0.3992,  0.3329, -0.1341],
        [-0.2701, -0.0760, -0.2185,  0.1983, -0.2576],
        [-0.3480,  0.0790, -0.0043, -0.0468,  0.2294]], requires_grad=True)
number: 5
Parameter containing:
tensor([ 0.2503, -0.3742, -0.4155], requires_grad=True)
```

为什么会输出六个网络参数？对于每一层网络的参数，由权重W和偏置bias构成的。

```python
optimizer = optim.Adam(model.parameters(), lr=config.lr) # 在调用优化器时，要穿入模型的参数
```

### named_parameters()

`named_parameters()`返回参数的名称及参数本身，可以按照参数名对一些参数进行处理。

```python
for k,v in net.named_parameters():
    print(k, "-", v.size())
```

返回的是键值对，`k`为参数的名称 ，`v`为参数本身。输出结果为：

```python
vgg.0.weight - torch.Size([64, 3, 3, 3])
vgg.0.bias - torch.Size([64])
vgg.2.weight - torch.Size([64, 64, 3, 3])
vgg.2.bias - torch.Size([64])
vgg.5.weight - torch.Size([128, 64, 3, 3])
vgg.5.bias - torch.Size([128])
vgg.7.weight - torch.Size([128, 128, 3, 3])
vgg.7.bias - torch.Size([128])
vgg.10.weight - torch.Size([256, 128, 3, 3])
vgg.10.bias - torch.Size([256])
vgg.12.weight - torch.Size([256, 256, 3, 3])
vgg.12.bias - torch.Size([256])
vgg.14.weight - torch.Size([256, 256, 3, 3])
vgg.14.bias - torch.Size([256])
vgg.17.weight - torch.Size([512, 256, 3, 3])
vgg.17.bias - torch.Size([512])
vgg.19.weight - torch.Size([512, 512, 3, 3])
vgg.19.bias - torch.Size([512])
vgg.21.weight - torch.Size([512, 512, 3, 3])
vgg.21.bias - torch.Size([512])
vgg.24.weight - torch.Size([512, 512, 3, 3])
vgg.24.bias - torch.Size([512])
vgg.26.weight - torch.Size([512, 512, 3, 3])
vgg.26.bias - torch.Size([512])
vgg.28.weight - torch.Size([512, 512, 3, 3])
vgg.28.bias - torch.Size([512])
```

参数名的命名规则`属性名称.参数属于的层的编号.weight/bias`。 这在`fine-tuning`的时候，给一些特定的层的参数赋值是非常方便的，这点在后面在加载预训练模型时会看到。



## 数据增强

## 基本结构：

```python
# 自定义数据集
class experimental_dataset(Dataset):
		# 注意在这里的参数要有transform
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item
# ---------------------------------------------
   # 自定义数据增强函数
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

x = torch.rand(8, 1, 2, 2)

train_data = experimental_dataset(x, transform)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

```



## 注意事项

**注意**：数据增强不会增加数据集的size，只是会从一个batch中随机挑取数据进行数据增强，所以数据集的size还是和原来一样

如果想增加数据集：```increased_dataset = torch.utils.data.ConcatDataset([transformed_dataset,original])```

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

What your `data_transforms['train']` does is:

- Randomly resize the provided image and randomly crop it to obtain a `(224, 224)` patch
- Apply or not a random horizontal flip to **this patch**, with a 50/50 chance
- Convert **it** to a `Tensor`
- Normalize the resulting `Tensor`, given the mean and deviation values you provided

What your `data_transforms['val']` does is:

- Resize your image to `(256, 256)`
- Center crop the **resized image** to obtain a `(224, 224)` patch
- Convert **it** to a `Tensor`
- Normalize the resulting `Tensor`, given the mean and deviation values you provided



## 数据集加载方式

- **class torch.utils.data.Dataset**: 一个抽象类， 所有其他类的数据集类都应该是它的子类。而且其子类必须重载两个重要的函数：len(提供数据集的大小）、getitem(支持整数索引)。

- **class torch.utils.data.TensorDataset**: 封装成tensor的数据集，每一个样本都通过索引张量来获得。

- **class torch.utils.data.ConcatDataset**: 连接不同的数据集以构成更大的新数据集。

- **class torch.utils.data.Subset(dataset, indices)**: 获取指定一个索引序列对应的子数据集。

- **class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)**: 数据加载器。组合了一个数据集和采样器，并提供关于数据的迭代器。

- **torch.utils.data.random_split(dataset, lengths)**: 按照给定的长度将数据集划分成没有重叠的新数据集组合。

- **class torch.utils.data.Sampler(data_source)**:所有采样的器的基类。每个采样器子类都需要提供 **iter** 方-法以方便迭代器进行索引 和一个 len方法 以方便返回迭代器的长度。

- **class torch.utils.data.SequentialSampler(data_source)**:顺序采样样本，始终按照同一个顺序。

- **class torch.utils.data.RandomSampler(data_source)**:无放回地随机采样样本元素。

- **class torch.utils.data.SubsetRandomSampler(indices)**：无放回地按照给定的索引列表采样样本元素。

- **class torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)**: 按照给定的概率来采样样本。

- **class torch.utils.data.BatchSampler(sampler, batch_size, drop_last)**: 在一个batch中封装一个其他的采样器。

- **class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)**:采样器可以约束数据加载进数据集的子集。

  

### SubsetRandomSampler

**class torch.utils.data.SubsetRandomSampler(indices)**：无放回地按照给定的索引列表采样样本元素。

```python
dataset = MyCustomDataset(my_path)
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# Usage Example:
num_epochs = 10
for epoch in range(num_epochs):
    # Train:   
    for batch_index, (faces, labels) in enumerate(train_loader):
      
```
