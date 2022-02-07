import os
import numpy as np
import torch
import json
from numpy import (array, unravel_index, nditer, linalg, random, subtract)

from torch.utils.data import Dataset


def make_dataset_modelnet10(mode, opt):
    dataset = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet10_numpy')


    f = open(os.path.join(DATA_DIR, 'modelnet10_shape_names.txt'))
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join(DATA_DIR, 'modelnet10_train.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test' == mode:
        f = open(os.path.join(DATA_DIR, 'modelnet10_test.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    for i, name in enumerate(lines):
        # locate the folder name
        folder = name[0:-5]
        file_name = name

        # get the label
        label = shape_list.index(folder)

        item = (os.path.join(DATA_DIR, folder, file_name + '.npy'), label)
        dataset.append(item)

    return dataset




def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def scale_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2/3, high=1.5, size=[3])
    #xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    scale_pointcloud = np.multiply(pointcloud, xyz1).astype('float32')
    return scale_pointcloud

def rotate_perturbation_point_cloud(data):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    #angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    angles = np.random.uniform(low=0, high=360, size=[3])
    angles = angles*np.pi/180
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_data = np.dot(data, R).astype(np.float32)

    return rotated_data


def rotate_point_cloud_z (data):
    """ Randomly rotate the point clouds by z axis.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.random.uniform(low=0, high=360)
    angles = angles*np.pi/180

    Rz = np.array([[np.cos(angles),-np.sin(angles),0],
                   [np.sin(angles),np.cos(angles),0],
                   [0,0,1]])

    rotated_data = np.dot(data, Rz)

    return rotated_data


def rotate_point_cloud_y (data):
    """ Randomly rotate the point clouds by z axis.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.random.uniform(low=0, high=360)
    angles = angles*np.pi/180

    Ry = np.array([[np.cos(angles), 0, np.sin(angles)],
                   [0, 1, 0],
                   [-np.sin(angles), 0, np.cos(angles)]])

    rotated_data = np.dot(data, Ry)

    return rotated_data


def random_dropout_pointcloud(pointcloud):

    N, C = pointcloud.shape
    dropout_ratio = float( np.random.random() * 0.1) # 0~0.875
    drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
    pointcloud[drop_idx.tolist(), 0:3] = pointcloud[0, 0:3]  # set to the first point
    # pointcloud[drop_idx.tolist(), 0:3] = [0,0,0]
    return pointcloud



class ModelNet_Loader(Dataset):
    def __init__(self, opt, num_points, partition='train'):
        super(ModelNet_Loader, self).__init__()

        self.opt = opt
        self.partition = partition
        self.num_points = num_points
        
        self.dataset = make_dataset_modelnet10(self.partition, opt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pc_np_file, class_id = self.dataset[index]
        data = np.load(pc_np_file)
        data = data[np.random.choice(data.shape[0], self.num_points, replace=False), :]
        pointcloud = data[:, 0:3]  # Nx3

        # augmentation
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud) # following DGCNN, will always use this augmentation
            
            if self.opt.drop_point:
                pointcloud = random_dropout_pointcloud(pointcloud)

            if self.opt.train_rot_y_perturbation:
                pointcloud = rotate_point_cloud_y(pointcloud)

            if self.opt.train_rot_all_perturbation:
                pointcloud = rotate_perturbation_point_cloud(pointcloud)

        # convert to tensor
        pointcloud = pointcloud.astype(np.float32)  # 3xN

        return pointcloud, class_id


class ShapeNetDataset(Dataset):
    def __init__(self,
                 opt,
                 root='./data/shapenetcore_partanno_segmentation_benchmark_v0',
                 npoints=2500,
                 classification=True,
                 class_choice=None,
                 split='train'):
        self.npoints = npoints
        self.opt = opt
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        # self.data_augmentation = data_augmentation
        self.split = split
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        print(self.cat)

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        # print(point_set.shape, seg.shape)

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True) # two pc will have less than 1024 points, so use replace=True
        # resample
        point_set = point_set[choice, :]

        if self.split == "train":
            point_set = translate_pointcloud(point_set)  # following DGCNN, will always use this augmentation

            if self.opt.drop_point:
                point_set = random_dropout_pointcloud(point_set)

            if self.opt.train_rot_y_perturbation:
                point_set = rotate_point_cloud_y(point_set)

            if self.opt.train_rot_all_perturbation:
                point_set = rotate_perturbation_point_cloud(point_set)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            seg = np.loadtxt(fn[2]).astype(np.int64)
            seg = seg[choice]
            seg = torch.from_numpy(seg)
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



