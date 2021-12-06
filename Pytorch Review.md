

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



### 1.5 pytorch 把tensor转换成int

直接在tensor变量的后面加.item()，就能把tensor类型转换成int类型。



## detach()函数的作用

当我们训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者只训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播。

Detach()返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad, 这样我们就会继续使用这个新的Variable进行计算，后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播。



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
