# Pytorch Review

## Pytorch very slow to convert list of numpy arrays into tensors!!

Code 1:

```
import torch
import numpy as np

a = [np.random.randint(0, 10, size=(7, 7, 3)) for _ in range(100000)]
b = torch.tensor(np.array(a))
```

And code 2:

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



## pytorch 把tensor转换成int

直接在tensor变量的后面加.item()，就能把tensor类型转换成int类型。

## 如何把sensor的float list转为int的list：

```python
labels -> sensor的float list
new = [t.cpu().detach().numpy().astype(np.int) for t in labels] 
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

## Pytorch数据读取```__getitem__```的顺序问题：

```def __getitem__(self,idx):```
idx的范围是从0到len-1（__len__的返回值）

但是如果采用了dataloader进行迭代，num_workers大于一的话，因为是多线程，所以运行速度不一样，这个时候如果在__getitem__函，数里输出idx的话，就是乱序的。但是实际上当线程数设置为1还是顺序的。

## Pytorch保存和加载训练模型

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

   

## Pytorch加载预训练模型

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