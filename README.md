# DRL-PARL
这是一个记录和分享强化学习代码和思路的Github仓库，包括基础概念、算法和应用案例。欢迎大家贡献代码和提出建议。
<div align="center">
  <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/RL%E6%A6%82%E8%A7%88.png">
</div>
- 笔记链接
:notebook: [点击这里查看我的笔记](https://onedrive.live.com/view.aspx?resid=587F20AF69B21397%21583&id=documents)  

### 一、表格型方法
#### Q-learning 简介
- Sarsa全称是state-action-reward-state'-action'，目的是学习特定的state下，特定action的价值Q，最终建立和优化一个Q表格，以state为行，action为列，根据与环境交互得到的reward来更新Q表格，更新公式为：
  
  <div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/Sarsa.png">
  </div>
- Sarsa在训练中为了更好的探索环境，采用ε-greedy方式来训练，有一定概率随机选择动作输出。

#### Q-learning 简介
- Q-learning采用Q表格的方式存储Q值（状态动作价值），决策部分采用ε-greedy方式增加探索。
- Q-learning与Sarsa的不同之处在于更新Q表格的方式。
  - Sarsa是on-policy的更新方式，先做出动作再更新。
  - Q-learning是off-policy的更新方式，更新learn()时无需获取下一步实际做出的动作next_action，并假设下一步动作是取最大Q值的动作。
- Q-learning的更新公式为：
  `Q(s, a) = Q(s, a) + α * [r + γ * maxQ(s', a') - Q(s, a)]`

### 二、基于神经网络方法
#### DQN简介
  表格型方法存储的状态数量有限，当面对围棋或机器人控制这类有数不清的状态的环境时，表格型方法在存储和查找效率上都受局限，DQN的提出解决了这一局限，使用神经网络来近似替代Q表格。
本质上DQN还是一个Q-learning算法，更新方式一致。为了更好的探索环境，同样的也采用**ε-greedy**方法训练。
在Q-learning的基础上，DQN提出了两个技巧使得Q网络的更新迭代更稳定。
- **经验回放 Experience Replay**：主要解决样本关联性和利用效率的问题。使用一个经验池存储多条经验s,a,r,s'，再从中随机抽取一批数据送去训练。
- **固定Q目标 Fixed-Q-Target**：主要解决算法训练不稳定的问题。复制一个和原来Q网络结构一样的Target Q网络，用于计算Q目标值。
#### DQN 算法流程
  <div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/DQN.png">
  </div>

### 三、基于策略梯度方法
#### Policy Gradient —— Reinforce简介
- 在强化学习中，有两大类方法，一种基于值（Value-based），一种基于策略（Policy-based）
  - Value-based的算法的典型代表为Q-learning和SARSA，将Q函数优化到最优，再根据Q函数取最优策略。
  - Policy-based的算法的典型代表为Policy Gradient，直接优化策略函数。

- 采用神经网络拟合策略函数，需计算策略梯度用于优化策略网络。
  - 优化的目标是在策略`π(s,a)`的**期望回报**：所有的轨迹获得的回报R与对应的轨迹发生概率p的加权和，当N足够大时，可通过**采样**N个Episode求平均的方式近似表达。
  <div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/sample.png">
  </div>

   - 优化目标对参数θ求导后得到策略梯度：
  <div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/gradient.png">
  </div>

#### Reinforce 算法流程
  <div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/Reinfroce_procedure.png">
  </div>
  
### 四、连续动作空间
#### DDPG 简介
- DDPG的提出动机其实是为了让DQN可以扩展到连续的动作空间。
- DDPG借鉴了DQN的两个技巧：**经验回放** 和 **固定Q网络**。
- DDPG使用策略网络直接输出确定性动作。
- DDPG使用了Actor-Critic的架构。
#### 策略网络和Q网络（Actor-Critic）
##### DQN->DDPG
<div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/AC-1.png">
</div>

##### DDPG 网络更新
<div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/AC-2.png">
</div>

##### DDPG 代码框架
<div align="center">
    <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/DDPG.png">
</div>
