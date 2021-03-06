import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nets import RL_net
from af_env import AF_env
from utils import *
from PIL import Image
from torchvision import transforms

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 5   # target update frequency
MEMORY_CAPACITY = 20
env = AF_env(range_size = 3, step_size = 2, source_folder = './data/tmp')
N_ACTIONS = env.n_actions
HIDDEN_SIZE = 256
IMG_SIZE = 224
TEST_INTER = 10


TRANSFORM = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = RL_net(HIDDEN_SIZE, N_ACTIONS), RL_net(HIDDEN_SIZE, N_ACTIONS)
        self.learn_step_counter = 0                                     # for target updating
        self.memory = memory(MEMORY_CAPACITY, IMG_SIZE) # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.transform = TRANSFORM

    def choose_action(self, file_name_s, is_test = False):
        x = self.transform(Image.open(file_name_s))
        x = x.unsqueeze(0)
        # input only one sample
        if np.random.uniform() < EPSILON or is_test :   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action

    def store_transition(self, file_name_s, a, r, file_name_s_):
        self.memory.store_transition(file_name_s, a, r, file_name_s_, transform = self.transform)

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

        b_s = torch.FloatTensor(batch_memory_s)
        b_a = torch.LongTensor(batch_memory_a)
        b_r = torch.FloatTensor(batch_memory_r)
        b_s_ = torch.FloatTensor(batch_memory_s_)

        # print ('b_s shape', b_s.shape)
        # print ('b_s_ shape', b_s_.shape)

        # q_eval w.r.t the action in experience
        gather_b_a = b_a.view([BATCH_SIZE, -1])
        q_eval = self.eval_net(b_s).gather(1, gather_b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r.view(BATCH_SIZE, 1) + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s, file_name_s = env.reset()
    ep_r = 0
    count = 0
    while True:
        env.render()
        a = dqn.choose_action(file_name_s)

        # take action
        s_, file_name_s_, r, done = env.step(a)

        # modify the reward
        # r =

        dqn.store_transition(file_name_s, a, r, file_name_s_)

        ep_r += r
        if dqn.memory.memory_counter > MEMORY_CAPACITY:
            print ('start to learn ... ')
            dqn.learn()
            count += 1
            # if done:
            #     print('Ep: ', i_episode,
            #           '| Ep_r: ', round(ep_r, 2))
        if count % TEST_INTER == 0 and count != 0:
            print ('start to test ... ')
            s_a = dqn.choose_action(file_name_s_, is_test = True)
            print ('current state is : ', s_)
            print ('current action is: ', s_a)

        if done:
            break
        s = s_
        file_name_s = file_name_s_
