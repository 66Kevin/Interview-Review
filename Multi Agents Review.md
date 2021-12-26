[TOC]

# Week2 Embedded Agents

Agents are embedded in an environment, meaning that an agent affects and is affected by the environment. An agent experiences its environment through sensors an acts on its environment through effectors. 

## 2.1 Math revision

- Power Set: The power set of a set A is the set of all subsets of A. It is written as $2^A$
  - $A=\{x∣0<x<7$ and $x$ is odd$\}$. A={1,3,5} hence $2^A$={∅,{1},{3},{5},{1,3},{1,5},{3,5},{1,3,5}}

## 2.2 Accessible and inaccessible environments

- An accessible environment is one in which the agent can obtain complete, accurate, up-to-date information about the environment’s state

- Most moderately complex environments (including, for example, the everyday physical world and the Internet) are inaccessible.

Example:

- Accessible: Chess
- Inacessible: Stock market

## 2.3 Deterministic and non-deterministic environments

- A deterministic environment is one in which every action has a single guaranteed effect — there is no uncertainty about the state that will result from performing an action.
- The physical world can to all intents and purposes be regarded as non-deterministic.

## 2.4 Static and dynamic environments

- A static environment is one in which the only changes to the environment are those caused by actions made by the agent.对环境的唯一更改是由agent执行的操作引起的更改。

- A dynamic environment is one that has other processes and/or agents operating on it, and so changes in ways beyond the agent’s control. 不仅仅是有一个agent在决策，其他agents在决策时会互相影响

## 2.5 Formal specification of an embedded agent

$Env = <E, e_0, \pi>$

## 2.6 Utility Functions

Utility functions allow an agent to understand how "good" an outcome is. 



# Week3:Deductive,reactive and hybrid reasoning agents

- Deductive reasoning agents, which have an explicitly model of the world represented in logical formulas and reason using logical deduction. 
  - Concurrent MetateM, a specific language for specifying deductive reasoning agents.

- Reactive agents, which do not have an explicit model of the world that they reason with, but instead their behaviour is driven by a set of "stimulus -> action" rules.
  - The subsumption architecture, probably the most well known reactive agent architecture.

- Hybrid agents, which combine a purely reactive layer with higher-level reasoning layers.
  - We'll look at TouringMachines and InteRRaP as two examples of these. 


## 3.1 Deductive Agents

### 3.1.1 Concurrent MetateM

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211225184017386.png" alt="image-20211225184017386" style="zoom:40%;" />

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211225184524543.png" alt="image-20211225184524543" style="zoom:40%;" />

E.g:

$student(you)$  $\upsilon$ $graduate(you)$ -- at some point in the future you will graduate, until then you will be a student. 直到毕业前你一直是学生

$$title(me,dr)$$ $S$ 𝑎𝑤𝑎𝑟𝑑𝑒𝑑(𝑚𝑒,𝑝ℎ𝑑) -- at some point in the past I was awarded a PhD and my title has been Dr ever since then. 从授予我PHD学位以后，我一直都是Dr

$¬𝑓𝑟𝑖𝑒𝑛𝑑𝑠(𝑚𝑒,𝑦𝑜𝑢)$ $W $ $𝑎𝑝𝑜𝑙𝑜𝑔𝑖𝑠𝑒(𝑦𝑜𝑢,𝑚𝑒)$ -- we will never be friends unless you apologise to me.

$title(you,dr)$ $Z$ $awarded(you,phd)$ -- if it's true that you've been awarded a PhD in the past, then your title has been Dr ever since then.

## 3.2 Reactive Agents

 不再使用像MetaM中那样的符号来表示行为，只是通过响应环境而产生的各种简单行为的相互作用产生智能行为。

### 3.2.1 subsumption architecture

subsumption architecture - the most famous reactive agent architecture

把行为组织成等级层次结构，等级层级结构中的低层可以抑制高层，层次越低优先度越高，**其主要思想就是层次越高表示的越抽象的行为**举例来说，有人希望移动机器，人有躲避障碍的行为。这显然要给躲避障碍物赋予一个高的优先级。

举例：火星车Steels' experiments with the Mars explorer system

局限性：

- 由于纯反应式 Agent 按照局部信息（即关于Agent当前状态的信息）做出央策，很难想像这种决策方法能考虑非局部信息
- 必须经过费力费时才能创造出纯反应式 Agent
- Agents可用的行为越多，理解或预测它们的行为就越困难，因为不同层之间的相互作用的动态变得太复杂而无法理解。

## 3.3 Hybrid Agents



## Week2: Practical Reasoning Agents

## 2.1 Delibration and Means-ends reasoning

- **Deliberation**: deciding what state of affairs the agent wants to achieve; the results of deliberation are referred to as the agent's **intentions**. ( **weighing up** the options available to me and **deciding** which ones to commit to as intentions.)

- **Means-ends reasoning**: deciding how to achieve these states of affairs. (Once I've decided on my intention, I need to work out what I'm going to do to try to achieve this.)
- **Intentions**: We have seen that the output of the agent's deliberation are **intentions**. If an agent has an intention X this means is intends to (try) to bring about the state of affairs X.

## 2.2 Resource bounded reasoning and calculative rationality

calculative rationality

E.g:

Is the following statement true or false?

An agent with the property of calculative rationality that is situated in a static environment is guaranteed to always make optimal decisions.

TRUE

If an agent has the property of calculative rationality, the decisions it makes are guaranteed to be optimal according to the information it had available when it started making its decision. If the environment is static, this means that the only way the world can change is if the agent itself performs some action. This means that the world is guaranteed not to change during the time the agent is deciding what to do, and so a decision that is optimal according to the information available when the decision making process started will still be optimal when the agent finishes making its decision.

## 2.3 BDI agents

BDI stands for **Beliefs, Desires, Intentions**

Based on an agent's beliefs, it has a set of desires. Once an agent commits to trying to achieve one of its desires, this desire becomes an intention. 

E.g: If my workload eases then I might decide it is feasible to visit my family and commit to this as an intention. Once I have committed to this as an intention, we expect (as discussed previously) that: I will invest some effort in trying to achieve this;

### 2.3.1 BDI V1

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211226025455604.png" alt="image-20211226025455604" style="zoom:40%;" />

- When an agent receives a new percept from the environment, it updates its beliefs to take this new information into account using its belief revision function: $brf()$.
- $options()$, which takes as input the agent's current beliefs and intentions and returns a set of desires;
- $filter()$ function, which takes as input the agent's current beliefs, desires and intentions and returns the "best" options to commit to as its new intentions.
- $plan()$ Function,  which takes the agent's current beliefs and intentions, and returns a plan for the agent to follow in order to try to achieve its intentions. 
- Means-ends reasoning is concerned with working out what the agent is going to do to try to achieve its intentions. This is what the $plan()$ function does: determines a plan the agent is going to employ to try to achieve its intentions.
- **Deliberation is a two step process**. First the agent identifies its options (line 6) and then the agent filters these to select its intentions (line 7). Deliberation therefore takes place over lines 6 and 7.

### 2.3.2 BDI V2

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211226030123046.png" alt="image-20211226030123046" style="zoom:40%;" />

Version 2 of our agent control loop allows the agent to execute one action from its plan at a time, pausing to observe the environment after each action and reconsider whether its plan is still appropriate based on its new beliefs.

- $empty(π)$ is true if and only if the plan π is empty.
- $ℎ𝑒𝑎𝑑(𝜋)$ returns the first action of the plan π.
- $𝑡𝑎𝑖𝑙(𝜋)$ returns the remainder of the plan π once the first action of 𝜋 has been removed.
- $𝑠𝑜𝑢𝑛𝑑(𝜋,𝐼,𝐵)$ is true if and only if, according to the agent's beliefs 𝐵, the agent can expect the plan 𝜋 to achieve its intentions 𝐼. 



### 2.3.3 BDI V3

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211226030927828.png" alt="image-20211226030927828" style="zoom:40%;" />

In version 2 of the control loop, the only time the agent ever stops to consider its intentions is when it has completed executing its plan. We will now adapt the control loop so that the agent is able to adjust its intentions if it is appropriate to do so. For this we need to introduce the following predicates.

- $succeeded(I,B)$ is true if and only if, based on its beliefs B, the agent believes it has achieved its intentions 𝐼.
- $impossible(I,B)$ is true if and only if, based on its beliefs B, the agent believes it is impossible to achieve its intentions 𝐼

If the agent believes that its intentions are either impossible or already achieved, there is no point in the agent persisting in trying to achieve those intentions. 

Agents using version 3 of the BDI control loop stop and deliberate again in order to reconsider their intentions after every action they execute (lines 15-16 of control loop version 3).



### 2.3.4 BDI V4 (Meta-level)

慎思的过程需要花费大量的时间，当一个agent在慎思时，周围的环境也在改变，可能会提出没有关系的新形成的意图

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211226031704202.png" alt="image-20211226031704202" style="zoom:40%;" />



-  $reconsider(I,B)$ is true if and only if, given its beliefs B, the agent believes it is appropriate to reconsider its intentions 𝐼.

There is an important implicit assumption here: the cost of determining whether **reconsider(I,B) is true must be less than the cost of the actual deliberation** process itself (i.e., performing functions options and filters). If this were not the case, then the agent might as well go ahead and do the deliberation, since this would be less costly than trying to decide whether it should deliberate or not.



E.g：

<img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211226033608068.png" alt="image-20211226033608068" style="zoom:50%;" />

agents慎思之后就会改变intention，因为deliberation是两部分组成（ **weighing up** the options available to me and **deciding** which ones to commit to as intentions）。当且仅当Agent选择镇思的时候，使Agent 改变意图，两数reconsider (…)的行为是最优的。对于
Agent 选择了慎思过程，但没有改交意图时，则在慎思过程中花费的努力就是浪费。同样，如果 Agent应该改变意图，但是没有能改变失败了，则花费在实现意图上的努力也是浪费的。因此1和4最优
