#-*- coding: utf-8 -*-
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class Model(parl.Model):
    """ 使用全连接网络.
    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init___()
        hid1_size = act_dim * 10
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, x):
        out = paddle.tanh(self.fc1(x))
        prob = F.softmax(self.fc2(out), axis=-1)
        return prob
