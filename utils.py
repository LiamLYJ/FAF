import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import re

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
