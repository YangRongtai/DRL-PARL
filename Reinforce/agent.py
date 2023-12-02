import parl
import paddle
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = int(prob.argmax())  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        """ 根据训练数据更新一次模型参数
        """
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)
        return float(loss)