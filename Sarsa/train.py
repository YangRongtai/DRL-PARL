import time
import gym
import numpy as np

from env.env import FrozenLakeWapper, CliffWalkingWapper
from Agent import SarsaAgent
from evaluate.evaluate import Evaluate

assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"


# 训练
def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()
    action = agent.sample(obs)

    while True:
        next_obs, reward, done, _ = env.step(action)  # 与环境进行交互
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作

        # 学习 predict target 更新Q表
        agent.learn(obs, action, reward, next_obs, next_action, done)

        obs = next_obs
        action = next_action
        total_reward += reward
        total_steps += 1

        if render:
            env.render()

        if done:
            break
    return total_reward, total_steps


# 测试
def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # e_greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        time.sleep(0.5)
        env.render()

        if done:
            print('test reward = %.1f' % (total_reward))
            break
    return total_reward


def train():
    # 定义环境
    env = gym.make("FrozenLake-v0", is_slippery=False)
    env = FrozenLakeWapper(env)

    # env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    # env = CliffWalkingWapper(env)

    # 实例化一个Agent
    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greedy=0.1
    )

    rewards = [] # 统计奖励
    is_render = False
    for episode in range(1000):
        ep_reward, ep_steps = run_episode(env=env, agent=agent, render=is_render)
        if episode % 10 == 0:
            reward_sum = 0
            step_sum = 0
            for i in range(20):
                ep_reward, ep_steps = run_episode(env=env, agent=agent, render=is_render)
                reward_sum += ep_reward
                step_sum += ep_steps

            test_result = reward_sum/20
            test_steps = step_sum/20
            rewards.append(test_result)
            print('Episode %s: steps = %s, reward = %.1f' % (episode, test_steps, test_result))

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False

    print("......Start testing ......")
    test_episode(env, agent)
    agent.Q = np.zeros((agent.obs_n, agent.act_n))

    return rewards


if __name__ == '__main__':
    print("第一次训练......")
    rewards1 = train()
    time.sleep(2)
    print("第二次训练......")
    rewards2 = train()

    time.sleep(2)
    print("开始评估......")
    Evaluate(rewards1, rewards2)
