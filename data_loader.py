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
        # print ('file_names: ', file_names)
        # print ('labels: ', labels)
        assert (len(labels) == len(file_names))
        images = []
        trans_back = transforms.ToTensor()
        for i in range(len(file_names)):
            image = Image.open(file_names[i]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = trans_back(image)
            images.append(image)
        images = torch.stack(images, 0)
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
    trans_back = transforms.ToTensor()
    data.sort(key=lambda x: len(x[1]), reverse=True)
    tmp_images, labels= zip(*data)
    tmp_images, labels = tmp_images[0], labels[0]
    images = []
    for item in tmp_images:
        images.append(trans_back(item))
    images = torch.stack(images, 0)
    # print ('image shape: ', images.shape)
    # print ('label shape: ', labels.shape)

    return images, labels


def get_loader(data_file, transform, batch_size, shuffle, num_workers):
    af = AF(data_file, transform)

    data_loader = torch.utils.data.DataLoader(dataset=af,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              )
    # data_loader = torch.utils.data.DataLoader(dataset=af,
    #                                           batch_size=batch_size,
    #                                           shuffle=shuffle,
    #                                           num_workers=num_workers,
    #                                           collate_fn=collate_fn)
    return data_loader
