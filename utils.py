import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import re
from PIL import Image

class memory(object):
    def __init__(self, memory_capacity, img_size):
        self.img_size = img_size
        self.memory_capacity = memory_capacity
        self.memory_counter = 0
        self.state_pool = np.zeros([memory_capacity, img_size, img_size, 3])
        self.state_pool_ = np.zeros([memory_capacity, img_size, img_size, 3])
        self.reward_pool = np.zeros([memory_capacity, 1])
        self.action_pool = np.zeros([memory_capacity, 1])

    def store_transition(self, file_name_s, a,r ,file_name_s_):
        index = self.memory_counter % self.memory_capacity
        state_img = np.asarray(Image.open(s), dytpe = np.int32)
        state_img_ = np.asarray(Image.open(s_), dytpe = np.int32)
        self.state_pool[index, :] = state_img
        self.state_pool_[index, :] = state_img_
        self.reward_pool[index,0] = r
        self.action_pool[index,0] = a
        self.memory_counter += 1



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    tmp = [ tryint(c) for c in re.split('([0-9]+)', s) ]
    if tmp[0] == '-':
        tmp[1] = -1 * tmp[1]
    return tmp[1]

def sort_nicely(l):
    l.sort(key=alphanum_key)


def test():
    batch_size = 3
    max_length = 3
    hidden_size = 2
    n_layers =1
    num_input_features = 1
    input_tensor = torch.zeros(batch_size,max_length,num_input_features)
    print (input_tensor.shape)
    x = input_tensor
    y = torch.stack((x,x),1)
    print (y)
    print (y.shape)
    raise
    input_tensor[0]= torch.FloatTensor([1,2,3]).view(3,-1)
    input_tensor[1] = torch.FloatTensor([4,5,0]).view(3,-1)
    input_tensor[2] = torch.FloatTensor([6,0,0]).view(3,-1)
    print(input_tensor)
    batch_in = Variable(input_tensor)
    seq_lengths = [3,2,2]
    print ('seq length: ', seq_lengths)
    pack = pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
    print (pack)
    print ('0:', pack[0])

if __name__ == '__main__':
    test()
