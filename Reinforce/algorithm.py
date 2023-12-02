import parl
import paddle
from paddle.distribution import Categorical
import paddle.nn.functional as F

class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr):
        """ Policy Gradient algorithm

           Args:
               model (parl.Model): policy的前向网络.
               lr (float): 学习率.
        """
        assert isinstance(lr, float)
        self.model = model
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters()
        )

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        prob = self.model(obs) # 获取输出动作概率
        # log_prob = Categorical(prob).log_prob(action) # 交叉熵
        # loss = paddle.mean(-1 * log_prob * reward)
        action_onehot = paddle.squeeze(
            F.one_hot(action, num_classes=prob.shape[1]),axis=1)
        log_prob = paddle.sum(paddle.log(prob + 1e-8) * action_onehot, axis=-1)
        reward = paddle.squeeze(reward, axis=1)
        loss = paddle.mean(-1 * log_prob * reward)

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        return loss
