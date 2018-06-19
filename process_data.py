import pickle
import os
import sys
import numpy as np
from data_loader import get_loader

def test(dump_name):
    files= []
    labels = []
    for i in range(5):
        file_name = './data/images/%d.jpg'%(i)
        files.append(file_name)
        labels.append(i)
    files_list = []
    labels_list = []
    files_list.append(files)
    labels_list.append(labels)
    for _ in range(11):
        files_list.append(files)
        labels_list.append(labels)
    data = {}
    data['files_list'] = files_list
    data['labels_list'] = labels_list

    with open(dump_name, 'bw' ) as f:
        pickle.dump(data, f)
        print ('dump done')

    data_loader = get_loader(dump_name, transform = None, batch_size=2, shuffle =
                             True, num_workers = 1)
    print ('len: ', len(data_loader))
    for i, (images, labels ) in enumerate(data_loader):
        print ('****************')
        print ('image shape', images.shape)
        print ('labels shape: ', labels.shape)

if __name__ == '__main__':
    test('tmp.pkl')
