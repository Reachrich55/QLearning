# QLearning
Q学习算法解决回合式任务
## 1. 问题描述
假设一幢建筑里面有5个房间，房间之间通过门相连（如图1所示）。将这五个房间按照从0至4进行编号，且建筑的外围可认为是一个大的房间，编号为5。将机器人置于2号房间，利用Q学习算法获得一条使机器人从该房间撤离到建筑物外围的最优路径。  

<div align="center">
  <img src="https://github.com/user-attachments/assets/b304dc08-0b02-46c9-904d-a4555507fab7" alt="image">
</div>

## 2. 问题分析
### 2.1 符号系统
$s$：描述环境的状态  
$s'$：下一个状态  
$a$：动作  
$\pi(a|s)$：策略  
$r(s,a,s')$：奖励  
$\tau$：决策轨迹  
$G(\tau)$：总回报  
$T$：决策回合  
$\alpha$：学习率  
$Q(s,a)$：状态动作值函数  
### 2.2 环境处理
本次实验采用了一个简单的建筑环境，由6个房间构成，其中0至4为内部房间，5为建筑外围。将房间结构转化为图结构（如图2所示），利于机器表示。  

<div align="center">
    <img src="https://github.com/user-attachments/assets/50438ad3-93fa-4630-85e0-cab162b37614" alt="image">
</div>

```python
# 定义房间连接关系
graph = {
    0: [4],
    1: [3, 5],
    2: [3],
    3: [1, 2, 4],
    4: [0, 3, 5],
    5: []  # 目标房间5没有通往其他房间的出口
}
```
### 2.3 问题求解
Q学习是强化学习的方法之一，因为策略和状态转移都有一定的随机性，所以每次试验得到的轨迹是一个随机序列，其收获的总回报也不一样。强化学习的目标是学习到一个策略序列来最大化期望回报 $G(\tau)$ 在本题中即找到一条能够到达外围获得最大奖赏的策略。  
因为环境中存在终止状态，当机器人撤离房间时任务结束。即此次任务的决策次数 $T$ 为有限次，目标函数为：  
$$\underset {\tau}{arg max}G(\tau) = \sum_{t=0}^{T-1}r_{t+1}$$  
在Q学习中，Q函数的估计方法为：  
$$Q(s,a)\leftarrow Q(s,a)+\alpha(r+\gamma maxQ(s',a')-Q(s,a))$$  
鉴于此次任务并非持续式任务，从而可以忽略掉折扣率 $\gamma$ 对 $Q(s,a)$ 的影响。  
故Q值的更新过程为：  
```python
# 更新Q值
Q[state, action] = Q[state, action] + alpha * (reward + np.max(Q[next_state, :]) - Q[state, action])
```
为了鼓励机器人找到最优撤离路径，定义如下奖励规则：  
•	到达目标房间（状态5）时，获得高奖励，设为10。  
•	每走一步未到达目标时，给予一个小的负奖励，设为-1，以鼓励机器人尽快找到出口。  
```python
# 定义奖励函数
def get_reward(state, next_state):
    if next_state == 5:
        return 10  # 到达目标房间
    else:
        return -1  # 未找到出口的惩罚
```
Q学习的学习过程设计如下：  
输入 状态空间 $S$，动作空间 $A$，学习率 $\alpha$  
1 repeat  
2    &emsp;初始化起始状态 $s=2$  
3    &emsp;repeat  
4    &emsp;在状态 $s$，选择动作 $a$  
5    &emsp;执行动作 $a$，得到奖励 $r$ 和新状态 $s'$  
6    &emsp;更新Q函数 $Q(s,a)\leftarrow Q(s,a)+\alpha(r+\gamma maxQ(s',a')-Q(s,a))$  
7    &emsp; $s\leftarrow s'$  
8    &emsp;until $s$为终止状态  
9 until $Q(s,a)$收敛  
输出 决策轨迹 $\tau = \sum_{t=0}^{T}\pi_{t+1} = \underset {a}{arg max}Q(s,a)$  
Q学习结束条件 $Q(s,a)$ 收敛可以通过多次重启学习过程使得状态空间中的每一个状态都被不同动作执行。具体实现如下：  
```python
# Q学习算法
for episode in range(num_episodes):
    state = 2  # 每轮从房间2开始
    finished = False

    while not finished:

        action = random.choice(graph[state])  # 随机选择一个动作

        # 执行动作并获得下一个状态和奖励
        next_state = action
        reward = get_reward(state, next_state)

        # 更新Q值
        Q[state, action] = Q[state, action] + alpha * (reward + np.max(Q[next_state, :]) - Q[state, action])

        # 更新状态
        state = next_state

        # 检查是否到达目标状态
        if state == 5:
            finished = True
```
