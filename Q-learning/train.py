import gym
import numpy as np
from env.env import CliffWalkingWapper, FrozenLakeWapper
from evaluate.evaluate import Evaluate
from Agent import QLearningAgent
import time

assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"


def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()
    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)

        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs

        total_reward += reward
        total_steps += 1

        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0

    obs = env.reset()
    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        time.sleep(0.1)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def train():
    # 定义环境
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)

    # 实例化agent
    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greedy=0.1
    )

    # 开始训练
    rewards = []  # 统计奖励
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

            test_result = reward_sum / 20
            test_steps = step_sum / 20
            rewards.append(test_result)
            print('Episode %s: steps = %s, reward = %.1f' % (episode, test_steps, test_result))
        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    # 训练结束，查看算法效果
    print("......Start testing ......")
    test_episode(env, agent)
    agent.Q = np.zeros((agent.obs_n, agent.act_n))

    return rewards


if __name__ == "__main__":
    print("第一次训练......")
    rewards1 = train()
    time.sleep(2)
    print("第二次训练......")
    rewards2 = train()

    time.sleep(2)
    print("开始评估......")
    Evaluate(rewards1, rewards2)
