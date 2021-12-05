# 基础算法Review

## 分治法

Main idea：把问题分解为若干个子问题，把子问题逐个解决，再组合到一起形成总问题的解

**实现方式**：

循环递归

在每一层递归上都有三个步骤：

			 1. 分解：将原问题分解为若干个规模较小，相对独立，与原问题形式相同的问题
			 1. 解决：若子问题规模较小且易于解决时，则直接解。否则，继续分解
			 1. 合并：将各子问题的解合并为原问题的解

**注意事项：边界条件，即求解问题的最小规模的判定**

例题：

![image-20211119152619196](/Users/kevin/Library/Application Support/typora-user-images/image-20211119152619196.png)

![image-20211119152922799](/Users/kevin/Library/Application Support/typora-user-images/image-20211119152922799.png)

最大子序要么全在中心点左边，要么在右边，要么跨中心。

跨中心情况，可以在中心点求左侧最大和右侧最大，最后加上中心点，即为跨中心情况的最大子序。

```cpp
class Solution{
public:
  int maxSubArray(vector<int> &nums){
    return find(nums, 0, nums.size()-1);
  }
  
  int find(vector<int> &nums, int start, int end){
    //边界条件
    if(start == end) return nums[start];
    if(start > end) return INT_MIN;
    
    //declare
    int left_max = 0, right_max = 0, ml = 0, mr = 0;
    int middle = (start + end)/2;
    
    //find max in left or right
    left_max = find(nums, start, middle-1);
    right_max = find(nums, middle+1, end);
    //find max through middle using Greedy algorithm
    for (int i=middle-1, sum=0; i>=start; --i){
      sum += nums[i];
      if(ml<sum) ml=sum;
    }
    for (int i=middle+1, sum=0; i<=end; ++i){
      sum += nums[i];
      if(mr<sum) mr=sum;
    }
    return max(max(left_max, right_max),ml+nums[middle]+mr);
  }
};
```

