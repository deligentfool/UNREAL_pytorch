import numpy as np
import cv2

def img_process(img, out_shape = (84, 84)):
    img = cv2.resize(src=img, dsize=out_shape)
    img = np.transpose(img, [2, 0, 1])
    return img

def calculate_batch_mean(img, batch_size):
    mean = []
    for i in range(int(img.shape[1] / batch_size)):
        mean.append([])
        for j in range(int(img.shape[2] / batch_size)):
            batch = img[:, i * batch_size: (i + 1) * batch_size, j * batch_size: (j + 1) * batch_size]
            mean[-1].append(np.mean(batch))
    return np.array(mean)

def calculate_batch_reward(observation, next_observation, batch_size=4):
    assert observation.shape[1] % batch_size == 0 and observation.shape[2] % batch_size == 0
    observation_mean = calculate_batch_mean(observation, batch_size)
    next_observation_mean = calculate_batch_mean(next_observation, batch_size)
    absolute_difference = np.abs(observation_mean - next_observation_mean)
    return absolute_difference

def clip_img(img, size):
    h_margin = int((img.shape[1] - size) / 2)
    v_margin = int((img.shape[2] - size) / 2)
    return img[:, h_margin: -h_margin, v_margin: -v_margin]

def one_hot_numpy(action_dim, action):
    return np.eye(action_dim)[list(action)]


def pull_and_push(optimizer, local_net, global_net, buffer, action_dim, batch_size, pc_weight, rp_weight, vr_weight):
    observations, actions, rewards, dones = buffer.sample_main()
    actions_one_hot = one_hot_numpy(action_dim, actions)
    loss = local_net.main_loss(observations, actions_one_hot, rewards, dones, actions)

    if buffer.pc_available(batch_size):
        observations, actions, rewards, batch_rewards, next_observations, next_actions, next_rewards, dones = buffer.sample_pc(batch_size)
        actions_one_hot = one_hot_numpy(action_dim, actions)
        next_actions_one_hot = one_hot_numpy(action_dim, next_actions)
        pc_loss = local_net.pc_loss(observations, actions_one_hot, rewards, batch_rewards, next_observations, next_actions_one_hot, next_rewards, dones, actions)
        loss = loss + pc_weight * pc_loss

    if buffer.rp_available(batch_size):
        p_batch, n_batch = buffer.sample_rp(batch_size)
        rp_loss = local_net.rp_loss(p_batch, n_batch)
        loss = loss + rp_weight * rp_loss

    if buffer.vr_available(batch_size):
        observations, actions, rewards, dones = buffer.sample_vr(batch_size)
        actions_one_hot = one_hot_numpy(action_dim, actions)
        vr_loss = local_net.vr_loss(observations, actions_one_hot, rewards, dones)
        loss = loss + vr_weight * vr_loss

    optimizer.zero_grad()
    loss.backward()
    for l_p, g_p in zip(local_net.parameters(), global_net.parameters()):
        g_p._grad = l_p.grad
    optimizer.step()
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print('{}  [episode: {}]  [reward: {:.2f}]'.format(name, global_ep.value, global_ep_r.value))
