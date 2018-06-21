import pickle
import os
import sys
import numpy as np
from PIL import Image
from data_loader import get_loader
import glob

def process(input_folder, save_path, range_size =2, step = 2, type = "*.jpg", size = 256):
# foucs_index : index to be the most clear one
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(input_folder):
        if dirs:
            continue
        files.sort()
        folder = root.split('/')[-1]
        focus_index = get_focus_index(root, type = type)
        save_folder = os.path.join(save_path, folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # left and right plus focus one
        for i in range(2*range_size +1):
            jump = (i - range_size) * step
            index = jump + focus_index
            target_file = os.path.join(save_folder, '%03d.jpg'%(i))
            source_file = os.path.join(root, files[index])
            command = 'cp %s %s'%(source_file, target_file)
            os.system(command)

def get_focus_index(folder, type = '*.jpg'):
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
        files.sort()
        for file in files:
            label = int(file.split('.')[0])
            image_name = os.path.join(root, file)
            img_files.append(image_name)
            labels.append(label)
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
    # test('tmp.pkl')
    process('./data/FAF_data','./data/tmp')
    make_dump('./data/tmp', './data/tmp.pkl', check_results = True)
