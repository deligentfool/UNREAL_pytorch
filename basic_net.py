import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_net(nn.Module):
    def __init__(self, observation_dim):
        super(conv_net, self).__init__()
        self.observation_dim = observation_dim

        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.observation_dim[0], 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_dim(), 256),
            nn.ReLU()
        )

    def fc_dim(self):
        tmp = torch.zeros(1, * self.observation_dim)
        return self.conv_layer(tmp).view(1, -1).size(1)

    def forward(self, observation):
        x = self.conv_layer(observation)
        x = x.view(x.size(0), -1)
        conv_feature = self.fc_layer(x)
        return conv_feature


class lstm_net(nn.Module):
    def __init__(self, action_dim):
        super(lstm_net, self).__init__()
        self.action_dim = action_dim

        self.lstm_layer = nn.LSTM(256 + self.action_dim + 1, 256, 1, batch_first=True)

    def forward(self, conv_feature, action, reward, hidden=None):
        conv_feature = torch.cat([conv_feature, action, reward], 1)
        conv_feature = conv_feature.unsqueeze(0)
        if not hidden:
            h0 = torch.zeros(conv_feature.size(0), 1, 256)
            c0 = torch.zeros(conv_feature.size(0), 1, 256)
            hidden = (h0, c0)
        lstm_feature, new_hidden = self.lstm_layer(conv_feature, hidden)
        return lstm_feature.squeeze(0), new_hidden


class policy_net(nn.Module):
    def __init__(self, action_dim):
        super(policy_net, self).__init__()
        self.action_dim = action_dim

        self.policy_layer = nn.Linear(256, self.action_dim)

    def forward(self, lstm_feature):
        prob = F.softmax(self.policy_layer(lstm_feature), 1)
        return prob


class value_net(nn.Module):
    def __init__(self):
        super(value_net, self).__init__()

        self.value_layer = nn.Linear(256, 1)

    def forward(self, lstm_feature):
        value = self.value_layer(lstm_feature)
        return value


class pixel_control_net(nn.Module):
    def __init__(self, action_dim):
        super(pixel_control_net, self).__init__()
        self.action_dim = action_dim

        self.deconv_fc_layer = nn.Sequential(
            nn.Linear(256, 32 * 9 * 9),
            nn.ReLU()
        )
        self.value_deconv_layer = nn.ConvTranspose2d(32, 1, 4, 2)
        self.advan_deconv_layer = nn.ConvTranspose2d(32, self.action_dim, 4, 2)

    def forward(self, lstm_feature):
        x = self.deconv_fc_layer(lstm_feature)
        x = x.view(x.size(0), 32, 9, 9)
        value = self.value_deconv_layer(x)
        advan = self.advan_deconv_layer(x)
        return value + advan


class reward_prediction_net(nn.Module):
    def __init__(self, stack_num=3):
        super(reward_prediction_net, self).__init__()
        self.stack_num = stack_num
        self.reward_prediction_layer = nn.Sequential(
            nn.Linear(256 * self.stack_num, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, conv_feature):
        score = self.reward_prediction_layer(conv_feature)
        return score