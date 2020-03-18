import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
from replay_buffer import replay_buffer
from unreal import unreal
from utils import img_process, clip_img, one_hot_numpy, pull_and_push, record
import numpy as np


class worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode_counter, global_reward, res_queue, name, max_episode, gamma, env_id, capacity, train_freq, n_step, stack_num, pc_weight, rp_weight, vr_weight, batch_size, observation_dim, entropy_weight):
        super(worker, self).__init__()
        self.name = 'w' + name
        self.global_episode_counter = global_episode_counter
        self.global_reward = global_reward
        self.res_queue = res_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.max_episode = max_episode
        self.gamma = gamma
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.env = self.env.unwrapped
        self.action_dim = self.env.action_space.n
        self.observation_dim = observation_dim
        self.entropy_weight = entropy_weight
        self.local_net = unreal(self.observation_dim, self.action_dim, self.gamma, self.entropy_weight)
        self.capacity = capacity
        self.train_freq = train_freq
        self.n_step = n_step
        self.stack_num = stack_num
        self.pc_weight = pc_weight
        self.rp_weight = rp_weight
        self.vr_weight = vr_weight
        self.batch_size = batch_size

        self.buffer = replay_buffer(self.capacity, self.train_freq, self.n_step, self.stack_num, self.gamma)
        self.last_action = torch.zeros(1, self.action_dim)
        self.last_reward = torch.zeros(1, 1)

    def run(self):
        total_step = 1
        while self.global_episode_counter.value < self.max_episode:
            obs = self.env.reset()
            obs = img_process(obs, self.observation_dim[1:])
            episode_reward = 0
            while True:
                if self.name == 'w0':
                    self.env.render()
                action = self.local_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), self.last_action, self.last_reward)
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = img_process(next_obs, self.observation_dim[1:])
                self.last_action = torch.FloatTensor(one_hot_numpy(self.action_dim, [action]))
                self.last_reward = torch.FloatTensor([reward]).unsqueeze(0)
                self.buffer.store(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_reward += reward
                if total_step % self.train_freq == 0 or done:
                    pull_and_push(self.optimizer, self.local_net, self.global_net, self.buffer, self.action_dim, self.batch_size, self.pc_weight, self.rp_weight, self.vr_weight)
                    self.buffer.clear_main()
                    if done:
                        record(self.global_episode_counter, self.global_reward, episode_reward, self.res_queue, self.name)
                        break
                total_step += 1
        self.res_queue.put(None)