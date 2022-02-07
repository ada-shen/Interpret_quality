import os
import numpy as np
import torch
import json

from torch.utils.data import Dataset
from tools.final_util import DATA_MODELNET_SHAPLEY_TEST, DATA_SHAPENET_SHAPLEY_TEST


def make_dataset_modelnet10(mode, opt):
    dataset = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet10_numpy')

    f = open(os.path.join(DATA_DIR, 'modelnet10_shape_names.txt'))
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join("misc", DATA_MODELNET_SHAPLEY_TEST),'r') # selected samples
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    # elif 'test' == mode: # this mode is not used when computing Shapley value
    #     f = open(os.path.join(DATA_DIR, 'modelnet10_test.txt'), 'r')
    #     lines = [str.rstrip() for str in f.readlines()]
    #     f.close()
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




class ModelNet_Loader_Shapley_test(Dataset):
    def __init__(self, opt, num_points, partition='train'):
        super(ModelNet_Loader_Shapley_test, self).__init__()

        self.opt = opt
        self.partition = partition
        self.num_points = num_points

        self.dataset = make_dataset_modelnet10(self.partition, opt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pc_np_file, class_id = self.dataset[index]
        data = np.load(pc_np_file)
        pointcloud = data[0:self.num_points, 0:3]  # Nx3

        # No augmentation when testing shapley value!
        # convert to tensor
        pointcloud = pointcloud.astype(np.float32)  # Nx3

        return pointcloud, class_id

def farthest_point_sample_np(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    # farthest = np.random.randint(0, N)
    farthest = 0
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids


class ShapeNetDataset_Shapley_test(Dataset):
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
        self.split = split
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join("misc", DATA_SHAPENET_SHAPLEY_TEST) # selected samples
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

        # choice = np.arange(0, len(point_set))
        # choice = np.random.choice(len(seg), self.npoints, replace=False)

        # we use FPS to ensure the sampled points distribute uniformly in the space
        choice = farthest_point_sample_np(point_set, self.npoints)
        # resample
        point_set = point_set[choice, :]

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array(cls).astype(np.int64)) # original: np.array([cls]) -> now: np.array(cls)

        if self.classification:
            return point_set, cls
        else:
            seg = np.loadtxt(fn[2]).astype(np.int64)
            seg = seg[choice]
            seg = torch.from_numpy(seg)
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



