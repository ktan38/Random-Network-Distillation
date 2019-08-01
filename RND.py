import random
from models import *
from collections import deque
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from IPython.display import clear_output

from obstacle_tower_env import ObstacleTowerEnv
import matplotlib
from matplotlib import pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

SEED = 1
BATCH_SIZE = 256
LR = 0.0003
UP_COEF = 0.025
EX_COEF = 2.0
IN_COEF = 1.0
UP_PROP = 0.25
GAMMA = 0.99
EPS = np.finfo(np.float).eps

# set device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)



m_losses = []
f_losses = []
losses = []

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

def learn(net, tgt_net, pred_net, rand_net, optimizer, rep_memory):
    global mean
    global std

    net.train()
    tgt_net.train()
    pred_net.train()
    rand_net.train()

    train_data = random.sample(rep_memory, BATCH_SIZE)

    dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        pin_memory=use_cuda
    )

    # double DQN
    for i, (s, a, r_ex, r_in, _s, d) in enumerate(dataloader):
        s_batch = s.to(device).float()
        s_batch = s_batch.permute(0, 3, 1, 2)
        a_batch = a.detach().to(device).long()
        _s_batch = _s.to(device).float()
        _s_batch = _s_batch.permute(0, 3, 1, 2)
        _s_norm = normalize_obs(_s.detach().cpu().numpy(), mean, std)
        _s_norm_batch = torch.tensor(_s_norm).to(device).float()
        _s_norm_batch = _s_norm_batch.permute(0, 3, 1, 2)
        r_ex_batch = r_ex.to(device).float()
        r_in_batch = r_in.to(device).float()
        r_batch = EX_COEF * 0.5 * r_ex_batch + IN_COEF * 0.5 * r_in_batch
        done_mask = 1. - d.to(device).float()

        _q_batch = net(_s_batch)
        _a_batch = torch.argmax(_q_batch, dim=1)
        pred_f = pred_net(_s_norm_batch)

        with torch.no_grad():
            _q_batch_tgt = tgt_net(_s_batch)
            action_space = _q_batch_tgt.shape[1]
            _q_best_tgt = _q_batch_tgt[range(BATCH_SIZE), _a_batch]
            rand_f = rand_net(_s_norm_batch)

        q_batch = net(s_batch)
        q_acting = q_batch[range(BATCH_SIZE), a_batch]

        # loss
        m_loss = ((r_batch + GAMMA * done_mask *_q_best_tgt) - q_acting).pow(2).mean()
        m_losses.append(m_loss)

        f_loss = (pred_f - rand_f).pow(2)
        mask = torch.rand(f_loss.shape[1]).to(device)
        mask = (mask < UP_PROP).to(device).float()
        f_loss = (f_loss * mask).sum() / mask.sum().clamp(min=1)
        f_losses.append(f_loss)

        loss = m_loss + f_loss
        print("loss is" + str(loss))
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def select_action(obs, tgt_net):
    tgt_net.eval()
    with torch.no_grad():
        state = torch.tensor([obs]).to(device).float()
        print(state.shape)
        state = state.permute(0, 3, 1, 2)
        q = target_net(state)
        action = torch.argmax(q)

    return action.item()


def get_norm_params(obs_memory):
    global obs_apace

    obses = np.ndarray(shape=(84, 84, 3, 0))
    obses = obses.tolist()
    for obs in obs_memory:
        for i in range(len(obs)):
            for j in range(len(obs[0])):
                for k in range(len(obs[0][0])):
                    obses[i][j][k].append(obs[i][j][k])

    obses = np.asarray(obses)
    mean = np.zeros(obs_space, np.float32)
    (84, 84, 3)
    std = np.zeros(obs_space, np.float32)
    for i, obs_ in enumerate(obses):
        for j in range(len(obs_)):
            for k in range(len(obs_[j])):

                mean[i][j][k] = np.mean(obs_[j][k])
                std[i][j][k] = np.std(obs_[j][k])

    return mean, std + EPS


def normalize_obs(obs, mean, std):
    norm_obs = (obs - mean) / std


    return np.clip(norm_obs, -5, 5)


def calculate_reward_in(pred_net, rand_net, obs):
    global mean
    global std

    norm_obs = normalize_obs(obs, mean, std)
    state = torch.tensor([norm_obs]).to(device).float()
    print(state.shape)
    state = state.permute(0, 3, 1, 2)
    with torch.no_grad():
        pred_obs = pred_net(state)
        rand_obs = rand_net(state)
        reward = (pred_obs - rand_obs).pow(2).sum()
        print("reward is" + str(reward))
        # clipped_reward = torch.clamp(reward, -1, 1)
        # print("clipped reward is" + str(clipped_reward))

    return reward.item()

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot():
    plt.close()
    clear_output(True)
    plt.figure(figsize=(16, 5))
    plt.subplot(131)
    plt.plot(rewards, alpha=0.5)
    plt.subplot(131)
    plt.plot(reward_eval)
    plt.title(f'Extrinsic Reward: '
              f'{reward_eval[-1]}')
    print("one")
    plt.subplot(132)
    plt.plot(rewards_in, alpha=0.5)
    plt.title('Intrinsic Reward')
    plt.subplot(133)
    plt.plot(losses, alpha=0.5)
    plt.title('Loss')
    print("two")
    plt.ioff()
    plt.show(block=False)
    print("three")
    plt.pause(0.001)



# make an environment
# env = gym.make('CartPole-v0')
# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('LunarLander-v2')
env = ObstacleTowerEnv('./obstacletower', retro=True, realtime_mode=False)


env.seed(SEED)
obs_space = env.observation_space.shape
action_space = env.action_space.n

# hyperparameter
n_episodes = 1000
learn_start = 1500
memory_size = 50000
update_frq = 1
use_eps_decay = True
epsilon = 1.0
eps_min = 0.001
decay_rate = 0.0001
n_eval = 100

# global values
init_steps = 0
total_steps = 0
learn_steps = 0
rewards = []
rewards_in = []
reward_eval = []
is_learned = False
is_solved = False
is_init_roll = True

# make four nerual networks
net = DuelingDQN(action_space).to(device)
target_net = deepcopy(net)
pred_net = PredictNet().to(device)
rand_net = RandomNet().to(device)

# make a optimizer
total_params = list(net.parameters()) + list(pred_net.parameters())
optimizer = torch.optim.Adam(total_params, lr=LR)

# make memory
rep_memory = deque(maxlen=memory_size)
obs_memory = deque(maxlen=learn_start)


# rollout
while True:
    print("in loop")
    obs = env.reset()
    done = False
    while not done:
#         env.render()
        action = env.action_space.sample()
        _obs, _, done, _ = env.step(action)
        obs_memory.append(_obs)
        obs = _obs
        init_steps += 1
        if init_steps == 100:
            mean, std = get_norm_params(obs_memory)
            obs_memory.clear()
            is_init_roll = False
            break
    if not is_init_roll:
        break


# play!
for i in range(1, n_episodes + 1):
    print("loop 2")
    obs = env.reset()
    done = False
    ep_reward = 0
    ep_reward_in = 0.
    stk = -1
    while not done:
        stk += 1
        print("Loop 3")
#         env.render()
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = select_action(obs, target_net)

        _obs, reward, done, _ = env.step(action)

        reward_in = calculate_reward_in(pred_net, rand_net, _obs)

        obs_memory.append(_obs)
        rep_memory.append((obs, action, reward, reward_in, _obs, done))

        obs = _obs
        total_steps += 1
        ep_reward += reward
        ep_reward_in += reward_in
        print("ep inreward is" + str(ep_reward_in))

        if use_eps_decay:
            epsilon -= epsilon * decay_rate
            epsilon = max(eps_min, epsilon)

        if total_steps % learn_start == 0:
            mean, std = get_norm_params(obs_memory)

        if len(rep_memory) >= learn_start:
            if len(rep_memory) == learn_start:
                print('\n====================  Start Learning  ====================\n')
                is_learned = True
            learn(net, target_net, pred_net, rand_net,
                optimizer, rep_memory)
            learn_steps += 1

        if learn_steps == update_frq:
            # target smoothing update
            for t, n in zip(target_net.parameters(), net.parameters()):
                t.data = UP_COEF * n.data + (1 - UP_COEF) * t.data
            learn_steps = 0
    if done:
        print("in done")
        rewards.append(ep_reward)
        rewards_in.append(ep_reward_in)
        reward_eval.append(
            np.mean(list(reversed(rewards))[: n_eval]).round(decimals=2))
        print("plot")
        #plot()
        episode_durations.append(stk + 1)
        plot()
#         print('{:3} Episode in {:5} steps, reward {:.2f}, reward_in {:.2f}'.format(
#             i, total_steps, ep_reward, ep_reward_in))
        print("done plotting")
        if len(rewards) >= n_eval:
            print("in if")
            if reward_eval[-1] >= 100000:
                print('\n{} is sloved! {:3} Episode in {:3} steps'.format(
                    "ObstacleTower", i, total_steps))
                torch.save(target_net.state_dict(),
                        f'../test/saved_models/{"ObstacleTower"}_ep{i}_clear_model_dddqn_r.pt')
                break
print("done")
env.close()

# if __name__ == '__main__':
#     main()
