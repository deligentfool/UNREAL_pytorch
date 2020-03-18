import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_net import *


class unreal(nn.Module):
    def __init__(self, observation_dim, action_dim, gamma, entropy_weight):
        super(unreal, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim =action_dim
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        self.conv_net = conv_net(self.observation_dim)
        self.lstm_net = lstm_net(self.action_dim)
        self.policy_net = policy_net(self.action_dim)
        self.value_net = value_net()
        self.pixel_control_net = pixel_control_net(self.action_dim)
        self.target_pixel_control_net = pixel_control_net(self.action_dim)
        self.reward_prediction_net = reward_prediction_net()
        self.rp_loss_func = torch.nn.BCELoss()
        self.hidden = None

    def main_loss(self, observations, actions_one_hot, rewards, dones, actions):
        observations = torch.FloatTensor(observations)
        actions_one_hot = torch.FloatTensor(actions_one_hot)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        actions = torch.LongTensor(actions).unsqueeze(1)

        conv_features = self.conv_net.forward(observations)
        lstm_features, _ = self.lstm_net.forward(conv_features, actions_one_hot, rewards)
        probs = self.policy_net.forward(lstm_features)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().unsqueeze(0)
        log_probs = probs.gather(1, actions)
        values = self.value_net.forward(lstm_features)

        R = self.calculate_returns(values, dones).unsqueeze(1)
        delta = R.detach() - values
        policy_loss = (- delta.detach() * log_probs - self.entropy_weight * entropy).mean()
        value_loss = delta.pow(2).mean()
        return policy_loss + value_loss

    def pc_loss(self, observations, actions_one_hot, rewards, batch_rewards, next_observations, next_actions_one_hot, next_rewards, dones, actions):
        observations = torch.FloatTensor(observations)
        actions_one_hot = torch.FloatTensor(actions_one_hot)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1)
        next_observations = torch.FloatTensor(next_observations)
        next_actions_one_hot = torch.FloatTensor(next_actions_one_hot)
        next_rewards = torch.FloatTensor(next_rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        actions = torch.LongTensor(actions).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        conv_features = self.conv_net.forward(observations)
        lstm_features, _ = self.lstm_net.forward(conv_features, actions_one_hot, rewards)
        q_aux = self.pixel_control_net.forward(lstm_features)

        next_conv_features = self.conv_net.forward(next_observations)
        next_lstm_features, _ = self.lstm_net.forward(next_conv_features, next_actions_one_hot, next_rewards)
        next_q_aux = self.target_pixel_control_net.forward(next_lstm_features)

        actions = actions.expand(actions.size(0), 1, q_aux.size(2), q_aux.size(3))
        q = q_aux.gather(1, actions).squeeze(0)
        next_q = next_q_aux.max(1)[0].detach().unsqueeze(1)
        dones = dones.expand_as(next_q)
        target_q = batch_rewards + self.gamma * (1 - dones) * next_q
        loss = (target_q - q).pow(2).mean()
        return loss

    def rp_loss(self, p_batch, n_batch):
        p_set = []
        for stack in p_batch:
            stack = torch.FloatTensor(stack)
            p_set.append(self.conv_net.forward(stack).view(1, -1))
        p = torch.cat(p_set, 0)

        n_set = []
        for stack in n_batch:
            stack = torch.FloatTensor(stack)
            n_set.append(self.conv_net.forward(stack).view(1, -1))
        n = torch.cat(n_set, 0)

        p_label = torch.FloatTensor(p.size(0), 1).fill_(1.0)
        n_label = torch.FloatTensor(n.size(0), 1).fill_(0.0)

        p_loss = self.rp_loss_func(self.reward_prediction_net.forward(p), p_label)
        n_loss = self.rp_loss_func(self.reward_prediction_net.forward(n), n_label)
        return (p_loss + n_loss) / 2

    def vr_loss(self, observations, actions_one_hot, rewards, dones):
        observations = torch.FloatTensor(observations)
        actions_one_hot = torch.FloatTensor(actions_one_hot)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        conv_features = self.conv_net.forward(observations)
        lstm_features, _ = self.lstm_net.forward(conv_features, actions_one_hot, rewards)
        values = self.value_net.forward(lstm_features)
        R = self.calculate_returns(values, dones).unsqueeze(1)
        loss = (R.detach() - values).pow(2).mean()
        return loss

    def calculate_returns(self, values, dones):
        R_list = []
        R = 0
        for i in reversed(range(values.size(0))):
            R = R * self.gamma * (1 - dones[i]) + values[i]
            R_list.append(R)
        R_list = list(reversed(R_list))
        return torch.cat(R_list, 0)

    def act(self, observation, last_action, last_reward):
        conv_feature = self.conv_net.forward(observation)
        lstm_feature, self.hidden = self.lstm_net.forward(conv_feature, last_action, last_reward, self.hidden)
        probs = self.policy_net.forward(lstm_feature)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.detach().item()