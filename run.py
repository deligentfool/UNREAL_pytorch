from worker import worker
from unreal import unreal
import gym
from shared_adam import SharedAdam
import torch.multiprocessing as mp

if __name__ == '__main__':
    env_id = 'BreakoutNoFrameskip-v4'
    gamma = 0.99
    max_episode = 10000
    capacity = 200
    train_freq = 20
    n_step = 4
    stack_num = 3
    pc_weight = 1.
    rp_weight = 1.
    vr_weight = 1.
    batch_size = 32
    observation_dim = (3, 84, 84)
    entropy_weight = 1e-4

    env = gym.make(env_id)
    action_dim = env.action_space.n
    global_net = unreal(observation_dim, action_dim, gamma, entropy_weight)
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4)
    global_episode_counter, global_reward, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [
        worker(
            global_net=global_net,
            optimizer=optimizer,
            global_episode_counter=global_episode_counter,
            global_reward=global_reward,
            res_queue=res_queue,
            name=str(i),
            max_episode=max_episode,
            gamma=gamma,
            env_id=env_id,
            capacity=capacity,
            train_freq=train_freq,
            n_step = n_step,
            stack_num=stack_num,
            pc_weight=pc_weight,
            rp_weight=rp_weight,
            vr_weight=vr_weight,
            batch_size=batch_size,
            observation_dim=observation_dim,
            entropy_weight=entropy_weight
        )
        for i in range(1)
    ]
    [worker.start() for worker in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [worker.join() for worker in workers]