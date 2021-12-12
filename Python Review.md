

#  Python Review

1. Lambda 与 map结合：

  ```python
  list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))  # 使用 lambda 匿名函数
  ```

   Output: [1, 4, 9, 16, 25]
  
2. tuple与list区别：

​		1、list是一种有序的集合，可以随时添加和删除其中的元素

​		2、访问list中的元素，索引从0开始，0为第一个元素，当索引超出范围会报错，索引不能越界，最后一个元素 的索引是len(num)-1

​		3、如果要取最后一个元素，除了计算索引位置，还可以用-1做索引直接取到最后一个元素

​		4、 使用append()可以在list的尾部添加元素

​		5、使用insert()可以将元素插入添加到指定位置

​		6、使用pop()可以删除list最后一个元素； 使用pop(i)其中i是索引号，可以删除指定位置的元素

**因为tuple不可变，所以代码更安全。如果可能，能用tuple代替list就尽量用tuple。**

## DataFrame

### set_index(): 设置index

```python
>>> df = pd.DataFrame({'month': [1, 4, 7, 10],
...                    'year': [2012, 2014, 2013, 2014],
...                    'sale': [55, 40, 84, 31]})
>>> df
   month  year  sale
0      1  2012    55
1      4  2014    40
2      7  2013    84
3     10  2014    31
```

**将索引设置为 ‘month’ 列：**

```python
>>> df.set_index('month')
       year  sale
month
1      2012    55
4      2014    40
7      2013    84
10     2014    31
```

### reset_index()：重设index

当我们对Dataframe对象进行数据清洗之后, 例如去掉含NaN的行之后, 发现行索引还是原来的行索引,

例子1:

```python
df = pd.DataFrame([('bird', 389.0),
...                    ('bird', np.nan),
...                    ('mammal', 80.5),
...                    ('mammal', np.nan)],
...                   columns=('class', 'max_speed'))
>>> df
         class  max_speed
0			    bird      389.0
1			    bird        Nan
2		    mammal       80.5
3			  mammal        NaN
```
在去掉Nan值后，index依然和以前的一样：

```python
df = df.dropna()
>>> df
         class  max_speed
0			    bird      389.0
2		    mammal       80.5
```

此时用reset_index()后：

```
df = df.dropna().reset_index(drop=True)
>>> df
         class  max_speed
0			    bird      389.0
1		    mammal       80.5
```

drop参数举例：

```python
df = pd.DataFrame([('bird', 389.0),
...                    ('bird', 24.0),
...                    ('mammal', 80.5),
...                    ('mammal', np.nan)],
...                   index=['falcon', 'parrot', 'lion', 'monkey'],
...                   columns=('class', 'max_speed'))
>>> df
         class  max_speed
falcon    bird      389.0
parrot    bird       24.0
lion    mammal       80.5
monkey  mammal        NaN
```

**重置索引时，会将旧索引添加为列，并使用新的顺序索引：**

```python
>>> df.reset_index()
    index   class  max_speed
0  falcon    bird      389.0
1  parrot    bird       24.0
2    lion  mammal       80.5
3  monkey  mammal        NaN
```

**我们可以使用drop参数来避免将旧索引添加为列：**

```python
>>> df.reset_index(drop=True)
    class  max_speed
0    bird      389.0
1    bird       24.0
2  mammal       80.5
3  mammal        NaN
```

### reindex()：新加列

```python
>>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
>>> df = pd.DataFrame({'http_status': [200, 200, 404, 404, 301],
...                   'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
...                   index=index)
>>> df
           http_status  response_time
Firefox            200           0.04
Chrome             200           0.02
Safari             404           0.07
IE10               404           0.08
Konqueror          301           1.00
```

**创建一个新索引并重新索引该数据框。默认情况下，将分配新索引中在数据框中没有对应记录的值NaN**

```python
>>> new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
...              'Chrome']
>>> df.reindex(new_index)
               http_status  response_time
Safari               404.0           0.07
Iceweasel              NaN            NaN
Comodo Dragon          NaN            NaN
IE10                 404.0           0.08
Chrome               200.0           0.02
```

**我们可以通过将值传递给关键字来填充缺少的值fill_value。因为索引不是单调增加或减少，所以我们不能使用关键字的参数 method来填充NaN值**

```python
>>> df.reindex(new_index, fill_value=0)
               http_status  response_time
Safari                 404           0.07
Iceweasel                0           0.00
Comodo Dragon            0           0.00
IE10                   404           0.08
Chrome                 200           0.02
```

```python
>>> df.reindex(new_index, fill_value='missing')
              http_status response_time
Safari                404          0.07
Iceweasel         missing       missing
Comodo Dragon     missing       missing
IE10                  404          0.08
Chrome                200          0.02
```

**我们还可以重新索引列**

```python
>>> df.reindex(columns=['http_status', 'user_agent'])
           http_status  user_agent
Firefox            200         NaN
Chrome             200         NaN
Safari             404         NaN
IE10               404         NaN
Konqueror          301         NaN
```

**或者我们可以使用**axis-style参数

```python
>>> df.reindex(['http_status', 'user_agent'], axis="columns")
           http_status  user_agent
Firefox            200         NaN
Chrome             200         NaN
Safari             404         NaN
IE10               404         NaN
Konqueror          301         NaN
```

**为了进一步说明中的填充功能 reindex，我们将创建一个索引单调递增的数据框（例如，日期序列）**

```python
>>> date_index = pd.date_range('1/1/2010', periods=6, freq='D')
>>> df2 = pd.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
...                    index=date_index)
>>> df2
            prices
2010-01-01   100.0
2010-01-02   101.0
2010-01-03     NaN
2010-01-04   100.0
2010-01-05    89.0
2010-01-06    88.0
```

**假设我们决定扩展数据框以覆盖更大的日期范围**

```python
>>> date_index2 = pd.date_range('12/29/2009', periods=10, freq='D')
>>> df2.reindex(date_index2)
            prices
2009-12-29     NaN
2009-12-30     NaN
2009-12-31     NaN
2010-01-01   100.0
2010-01-02   101.0
2010-01-03     NaN
2010-01-04   100.0
2010-01-05    89.0
2010-01-06    88.0
2010-01-07     NaN
```

默认情况下，原始数据帧中没有值的索引条目（例如，`“2009-12-29”`）用填充`NaN`。如果需要，我们可以使用几个选项之一来填写缺失值

例如，要向后传播最后一个有效值以填充`NaN`值，请`bfill`作为参数传递给`method`关键字。

```python
>>> df2.reindex(date_index2, method='bfill')
            prices
2009-12-29   100.0
2009-12-30   100.0
2009-12-31   100.0
2010-01-01   100.0
2010-01-02   101.0
2010-01-03     NaN
2010-01-04   100.0
2010-01-05    89.0
2010-01-06    88.0
2010-01-07     NaN
```

注意： `NaN`任何值传播方案都不会填充原始数据帧中的值（索引值为`2010-01-03`）。这是因为在重新索引时进行填充不会查看数据帧值，而只会比较原始索引和所需索引。如果您确实想填写NaN原始数据框中存在的值，请使用该`fillna()`



### Concat()



