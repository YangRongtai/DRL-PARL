# DRL-PARL
这是一个记录和分享强化学习代码和思路的Github仓库，包括基础概念、算法和应用案例。欢迎大家贡献代码和提出建议。
<div align="center">
  <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/RL%E6%A6%82%E8%A7%88.png">
</div>

<H3>表格型方法——Sarsa</H3>
<H4>【Sarsa 简介】</H4>
    Sarsa全称是state-action-reward-state'-action'，目的是学习特定的state下，特定action的价值Q，最终建立和优化一个Q表格，以state为行，action为列，根据与环境交互得到的reward来更新Q表格，更新公式为：
<div align="center">
  <img src="https://github.com/YangRongtai/DRL-PARL/blob/master/data/Sarsa.png">
</div>
    Sarsa在训练中为了更好的探索环境，采用ε-greedy方式来训练，有一定概率随机选择动作输出。
