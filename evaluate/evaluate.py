#### 绘图分析
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Evaluate(rewards1, rewards2):
    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    rewards = np.vstack((rewards1, rewards2))  # 合并为二维数组
    df = pd.DataFrame(rewards).melt(var_name='episode', value_name='reward')

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x="episode", y="reward", data=df, color='blue', linestyle='--')  # DarkOrchid
    ax.set_title('Rewards over Episodes')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.legend(labels=['Reward'])
    plt.show()
