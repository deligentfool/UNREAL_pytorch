import numpy as np
from collections import deque
import random
from utils import calculate_batch_reward, one_hot_numpy, clip_img


class replay_buffer(object):
    def __init__(self, capacity, train_freq, n_step, stack_num, gamma):
        self.capacity = capacity
        self.train_freq = train_freq
        self.n_step = n_step
        self.stack_num = stack_num
        self.gamma = gamma

        self.main_buffer = deque(maxlen=self.train_freq)
        self.pixel_control_buffer = deque(maxlen=self.capacity)
        self.p_reward_prediction_buffer = deque(maxlen=self.capacity)
        self.n_reward_prediction_buffer = deque(maxlen=self.capacity)
        self.value_repeat_buffer = deque(maxlen=self.capacity)

        self.pc_tmp_buffer = deque(maxlen=self.n_step)
        self.rp_tmp_buffer = deque(maxlen=self.stack_num+1)

    def store_pc(self):
        if len(self.pc_tmp_buffer) == self.n_step:
            observation, action, reward = self.pc_tmp_buffer[0][: 3]
            batch_reward, next_observation, done = self.pc_tmp_buffer[-1][-3:]
            for _, _, _, batch_rew, next_obs, do in reversed(list(self.pc_tmp_buffer)[: -1]):
                batch_reward = self.gamma * batch_reward * (1 - do) + batch_rew
                next_observation, done = (next_obs, do) if do else (next_observation, done)
            self.pixel_control_buffer.append([observation, action, reward, batch_reward, next_observation, done])

    def store_rp(self):
        if len(self.rp_tmp_buffer) == self.stack_num + 1:
            observation_stack = []
            for traj in list(self.rp_tmp_buffer)[: -1]:
                observation_stack.append(traj[0])
            if self.rp_tmp_buffer[-1][1] > 0:
                self.p_reward_prediction_buffer.append(np.concatenate(observation_stack, 0))
            else:
                self.n_reward_prediction_buffer.append(np.concatenate(observation_stack, 0))

    def store(self, observation, action, reward, next_observation, done):
        clip_observation = clip_img(observation, 80)
        clip_next_observation = clip_img(observation, 80)
        batch_reward = calculate_batch_reward(clip_observation, clip_next_observation, batch_size=4)
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.main_buffer.append([observation, action, reward, next_observation, done])
        self.rp_tmp_buffer.append([observation, action, reward, next_observation, done])
        self.pc_tmp_buffer.append([observation, action, reward, batch_reward, next_observation, done])
        self.rp_tmp_buffer.append([observation, reward])
        self.store_pc()
        self.store_rp()
        self.value_repeat_buffer.append([observation, action, reward, next_observation, done])

        if done:
            self.rp_tmp_buffer.clear()

    def sample_main(self):
        observations, actions, rewards, next_observations, dones = zip(* self.main_buffer)
        return np.concatenate(next_observations, 0), actions, rewards, dones

    def sample_pc(self, batch_size):
        start_idx = random.choice(list(range(len(self.pixel_control_buffer) - batch_size - 1)))
        batch = list(self.pixel_control_buffer)[start_idx: start_idx + batch_size + 1]
        observations, actions, rewards, batch_rewards, next_observations, dones = zip(* batch)
        current_observations = next_observations[: -1]
        current_actions = actions[: -1]
        current_rewards = rewards[: -1]
        current_batch_rewards = batch_rewards[: -1]
        next_observations = next_observations[1:]
        next_actions = actions[1:]
        next_rewards = rewards[1:]
        dones = dones[: -1]
        return np.concatenate(current_observations, 0), current_actions, current_rewards, current_batch_rewards, np.concatenate(next_observations, 0), next_actions, next_rewards, dones

    def sample_rp(self, batch_size):
        p_batch = random.sample(self.p_reward_prediction_buffer, batch_size)
        n_batch = random.sample(self.n_reward_prediction_buffer, batch_size)
        return p_batch, n_batch

    def sample_vr(self, batch_size):
        start_idx = random.choice(list(range(len(self.value_repeat_buffer) - batch_size)))
        batch = list(self.value_repeat_buffer)[start_idx: start_idx + batch_size]
        observations, actions, rewards, next_observations, dones = zip(* batch)
        return np.concatenate(observations, 0), actions, rewards, dones

    def clear_main(self):
        self.main_buffer.clear()

    def pc_available(self, batch_size):
        return batch_size <= len(self.pixel_control_buffer)

    def rp_available(self, batch_size):
        return batch_size <= len(self.n_reward_prediction_buffer) and batch_size <= len(self.p_reward_prediction_buffer)

    def vr_available(self, batch_size):
        return batch_size <= len(self.value_repeat_buffer)
