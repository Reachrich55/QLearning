import numpy as np
import random
import matplotlib.pyplot as plt

# 定义参数
alpha = 0.1  # 学习率
num_episodes = 1000  # 训练回合数

# 定义房间连接关系
graph = {
    0: [4],
    1: [3, 5],
    2: [3],
    3: [1, 2, 4],
    4: [0, 3, 5],
    5: []  # 目标房间5没有通往其他房间的出口
}

# 初始化Q表
Q = np.zeros((6, 6))

# 存储每个回合的Q值变化，用于绘制收敛图
Q_values_over_time = []

# 定义奖励函数
def get_reward(state, next_state):
    if next_state == 5:
        return 10  # 到达目标房间
    else:
        return -1  # 未找到出口的惩罚


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

    if episode % 10 == 0:
            Q_values_over_time.append(Q.copy())

# 输出Q表
print("训练完成后的Q表：")
print(Q)


# 从2开始寻找最优路径
def find_optimal_path(start, goal=5):
    state = start
    path = [state]
    while state != goal:
        action = np.argmax(Q[state, :])
        path.append(action)
        state = action
    return path


optimal_path = find_optimal_path(2)
print("最优路径：", optimal_path)

# 绘制Q值更新趋势
plt.figure(figsize=(12, 6))
for state in range(6):
    for action in graph[state]:
        q_values = [Q[state, action] for Q in Q_values_over_time]
        plt.plot(q_values, label=f"Q[{state}, {action}]")
plt.xlabel("Episode (x10)")
plt.ylabel("Q-Value")
plt.title("Q-Value update trend line")
plt.legend()
plt.show()
