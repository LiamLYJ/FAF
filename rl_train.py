import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nets import RL_net
from af_env import AF_env
from utils import *
from PIL import Image

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = AF_env(range_size = 3, step_size = 2, source_folder = './data/tmp')
N_ACTIONS = env.n_actions
HIDDEN_SIZE = 256
IMG_SIZE = 256

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = RL_net(HIDDEN_SIZE, N_ACTIONS), RL_net(HIDDEN_SIZE, N_ACTIONS)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = memory(MEMORY_CAPACITY, IMG_SIZE) # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, file_name_s):
        x = np.asarray(Image.open(file_name_s), np.int32)
        x = torch.Tensor(x)
        x = x.unsqueeze(0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action

    def store_transition(self, file_name_s, a, r, file_name_s_):
        self.memory.store_transition(file_name_s, a, r, file_name_s_)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        batch_memory_s = self.memory.state_pool[sample_index, :]
        batch_memory_s_ = self.memory.state_pool_[sample_index, :]
        batch_memory_a = self.memory.action_pool[sample_index, 0]
        batch_memory_r = self.memory.reward_pool[sample_index, 0]

        b_s = torch.LongTensor(batch_memory_s)
        b_a = torch.LongTensor(batch_memory_a)
        b_r = torch.FloatTensor(batch_memory_r)
        b_s_ = torch.LongTensor(batch_memory_s_)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s, file_name_s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(file_name_s)

        # take action
        s_, file_name_s_, r, done = env.step(a)

        # modify the reward
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        dqn.store_transition(file_name_s, a, r, file_name_s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
        file_name_s = file_name_s_
