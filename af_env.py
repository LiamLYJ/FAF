import numpy as np
import time
import sys
import os


class AF_env(object):
    def __init__(self, range_size, step_size, source_folder):
        self.action_space = ['freeze', 'small_left', 'small_right', 'big_left', 'big_right']
        self.n_actions = len(self.action_space)
        self.step_size = step_size
        self.range_size = range_size
        self.min_state = -1 * range_size
        self.max_state = range_size
        self.folder_num = 0
        for root, dirs, files in os.walk(source_folder):
            if root == source_folder:
                self.folder_num = len(dirs)
        self.source_folder = source_folder
        self.positions = np.arange(-1 * range_size, range_size + 1)
        self._build()


    def _build(self):
        while True:
            self.state = np.random.choice(self.positions)
            # init state shoud not be focus postion
            if self.state != 0:
                break


    def reset(self):
        self._build()
        time.sleep(0.5)
        file_name = self.get_file_name(self.state)
        # return observation
        return self.state, file_name


    def get_file_name(self, s):
        select_folder = np.random.choice(self.folder_num)
        file_name = os.path.join(self.source_folder, '%05d'%(select_folder), '%d.png'%(s))
        while (not os.path.isfile(file_name)):
            # print ('not exists , get again')
            select_folder = np.random.choice(self.folder_num)
            file_name = os.path.join(self.source_folder, '%05d'%(select_folder), '%d.png'%(s))
        return file_name


    def step(self, action):
        s = self.state
        if action == 0:   # freeze
            s_ = s
        elif action == 1:   # small left
            s_ = max(self.min_state, s - self.step_size)
        elif action == 2:   # small right
            s_ = min(self.max_state, s + self.step_size)
        elif action == 3:   # big_left
            s_ = max(self.min_state, s - 2 * self.step_size)
        elif action == 4:   # big_right
            s_ = min(self.max_state, s + 2 * self.step_size)
        else:
            print ('action not recognized')
            assert False

        self.state = s_
        # reward function
        if s_ == 0:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        file_name = self.get_file_name(s_)
        return s_, file_name, reward, done


    def render(self):
        time.sleep(0.1)


def update(env):
    for t in range(2):
        s, file_name = env.reset()
        print ('state: ', s)
        print ('file_name: ', file_name)
        print ('***********************')
        while True:
            env.render()
            a = np.random.choice(5)
            s, file_name, r, done = env.step(a)
            print ('state: ', s)
            print ('reeward: ', r)
            print ('done: ', done)
            print ('file_name: ', file_name)
            print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            if done:
                break


if __name__ == '__main__':
    env = AF_env(range_size = 3, step_size = 2, source_folder = './data/tmp')
    update(env)
