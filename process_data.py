import pickle
import os
import sys
import random
import numpy as np
from PIL import Image
from data_loader import get_loader
import glob
from utils import *

def get_random_roi(img, size, min_value = 30, max_value = 240, rate = 0.4):
    height, width, _ = np.array(img).shape
    while (True):
        # 0.2, 0.57, 0.2 0.46 is setted manuly
        # position_y, position_x = random.uniform(0.2, 0.57), random.uniform(0.2, 0.46)
        position_y, position_x = random.uniform(0.1, 1 - size / height), random.uniform(0.1, 1 - size / width)
        start_y = int(position_y * height)
        start_x = int(position_x * width)
        if (start_y + size) < height and (start_x + size) < width:
            roi = img.crop((start_y, start_x, start_y + size, start_x + size))
            roi_array = np.array(roi)
            total_pixel = size * size * 3
            good_pixel = ((min_value < roi_array) & (roi_array < max_value)).sum()
            # print ('good: ', good_pixel)
            # print ('rate: ', good_pixel / total_pixel)
            # print ('start_y:' , start_y)
            # print ('start_x' , start_x)
            if (good_pixel / total_pixel > rate):
                return start_y, start_x


def cut_roi(input_folder, save_path, per_samples = 10, type = '.png', size = 224):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    count = 0
    for root, dirs, files in os.walk(input_folder):
        if dirs:
            continue
        # print ('root: ', root)
        # print ('dirs: ', dirs)
        # print ('files: ', files)
        files.sort()
        for i in range(per_samples):
            for j,file in enumerate(files):
                cur_save_path = os.path.join(save_path, '%02d_%03d'%(count, i))
                if not os.path.exists(cur_save_path):
                    os.mkdir(cur_save_path)
                img = Image.open(os.path.join(root, file))
                if j == 0:
                    start_y, start_x = get_random_roi(img, size)
                roi = img.crop((start_y, start_x, start_y + size, start_x + size))
                roi.save(os.path.join(cur_save_path, '%03d%s'%(j,type)))
        count += 1


def process(input_folder, save_path, range_size =2, step = 5, type = "*.png", size = 224, remove_flag = None):
    # foucs_index : index to be the most clear one
    count = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(input_folder):
        if dirs:
            continue
        files.sort()
        folder = '%05d'%(count)
        focus_index = get_focus_index(root, type = type)
        save_folder = os.path.join(save_path, folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # left and right plus focus one
        for i in range(focus_index - range_size, focus_index + range_size + 1):
            jump = (i - focus_index) * step
            index = jump + focus_index
            target_file = os.path.join(save_folder, '%d.png'%(i-focus_index))
            if index >= 0 and index < len(files):
                source_file = os.path.join(root, files[index])
            else :
                if remove_flag is None:
                    print ('in %s not enough roi, skip %d'%(save_folder, (i - focus_index)))
                    continue
                else:
                    remove_flag = True
                    print ('contain not enough, delet this forder')
                    break
            command = 'cp %s %s'%(source_file, target_file)
            os.system(command)
        count += 1
        if remove_flag:
            command = 'rm -rf %s'%(save_folder)
            print (save_folder)
            os.system(command)
            remove_flag = False


def get_focus_index(folder, type = '*.png'):
    focus_index = 0
    count = 0
    sharpness = np.finfo(float).eps
    file_list = glob.glob(os.path.join(folder, type))
    file_list.sort()
    for file_name in file_list:
        im = Image.open(file_name).convert('L') # to grayscale
        array = np.asarray(im, dtype=np.int32)
        gy, gx = np.gradient(array)
        gnorm = np.sqrt(gx**2 + gy**2)
        cur_sharpness = np.average(gnorm)
        sharpness = max(sharpness, cur_sharpness)
        focus_index = focus_index if (sharpness > cur_sharpness) else count
        count += 1
    return focus_index

def make_dump(input_path, dump_name, check_results = False):
    img_files_list = []
    labels_list = []
    for root, dirs, files in os.walk(input_path):
        if dirs:
            continue
        img_files = []
        labels = []
        # files.sort()
        # sort by numer
        sort_nicely(files)
        # print (files)
        for label_index, file in enumerate(files):
            image_name = os.path.join(root, file)
            img_files.append(image_name)
            labels.append(label_index)
        img_files_list.append(img_files)
        labels_list.append(labels)
    data = {}
    data['files_list'] = img_files_list
    data['labels_list'] = labels_list
    with open(os.path.join(dump_name), 'wb') as f:
        pickle.dump(data, f)
        print ('dump done')

    if check_results:
        data_loader = get_loader(dump_name, transform = None, batch_size=2, shuffle =
                                 True, num_workers = 1)
        print ('len: ', len(data_loader))
        for i, (images, labels ) in enumerate(data_loader):
            print ('****************')
            print ('image shape', images.shape)
            print ('labels shape: ', labels.shape)


def test(dump_name):
    files= []
    labels = []
    for i in range(5):
        file_name = './data/images/%d.png'%(i)
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
    # test('tmp.pkl')
    # cut_roi('/Users/lyj/Desktop/af_data/', './data/af_data')
    process('./data/af_data','./data/tmp', range_size = 3, step = 6)
    # process('./data/af_data','./data/tmp', range_size = 3, step = 6, remove_flag = False)
    # make_dump('./data/tmp', './data/tmp.pkl', check_results = True)
