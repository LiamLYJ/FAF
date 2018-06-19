import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image


class AF(data.Dataset):
    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.transform = transform

        # data[0]: list of a sequence of file name 
        # data[1]: list of a sequece of positions
        with open(data_file, 'rb') as handle:
            self.data = pickle.load(handle)
            self.files_list = self.data['files_list']
            self.labels_list = self.data['labels_list']
            

    def __getitem__(self, index):
        file_names = self.files_list[index]
        labels = self.labels_list[index]
        print ('file_names: ', file_names)
        print ('labels: ', labels)
        assert (len(labels) == len(file_names))
        images = []
        for i in range(len(file_names)):
            image = Image.open(file_names[i]).convert('RGB')
            tmp = np.array(images)
            print ('tmp: ', tmp)
            print ('file_names: ', file_names[i])
            if self.transform is not None:
                image = self.transform(image)
                image = torch.Tensor(image)
            images.append(image)
        labels = torch.Tensor(labels)

        return images, labels 

    def __len__(self):
        return (len(self.files_list))


def collate_fn(data):
    # data: images, labels 
    # args:
        # iamge of shape (sequence, 3, 256, 256)
        # lables : (sequence,1)
    # return:
        # images of shape : (batch_size, sequence, 3, 256,256) 
        # targerts: (batch_size, sequence)
        # lengths: vakud sequence lengths

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, labels= zip(*data)
    print ('images: ', images)
    print ('labels: ', labels)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    # images = torch.stack(images, 0)
    print ('yes once')

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in labels]
    targets = torch.zeros(len(labels), max(lengths)).long()
    for i, cap in enumerate(labels):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_loader(data_file, transform, batch_size, shuffle, num_workers):
    af = AF(data_file, transform)
    data_loader = torch.utils.data.DataLoader(dataset=af, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,)

    # data_loader = torch.utils.data.DataLoader(dataset=af, 
                                              # batch_size=batch_size,
                                              # shuffle=shuffle,
                                              # num_workers=num_workers,
                                              # collate_fn=collate_fn)
    return data_loader

