# DRL-PARL
这是一个记录和分享强化学习代码和思路的Github仓库，包括基础概念、算法和应用案例。欢迎大家贡献代码和提出建议。
<div align="center">
  <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/RL%E6%A6%82%E8%A7%88.png">
</div>

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
