import os
import argparse
import torch
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test
import numpy as np
from torch.utils.data import DataLoader
from tools.final_util import set_random, NUM_POINTS, NUM_REGIONS, SHAPENET_CLASS


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long,device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long,device=xyz.device)    # batch_indices shape is torch.Size([B])
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Bx1x3
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def save_fps(args):
    if args.dataset == "modelnet10":
        data_loader = DataLoader(ModelNet_Loader_Shapley_test(args, partition='train', num_points=args.num_points),num_workers=8,
                                  batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "shapenet":
        data_loader = DataLoader(ShapeNetDataset_Shapley_test(args, split='train', npoints=args.num_points,
                                                  class_choice=SHAPENET_CLASS, classification=True), num_workers=8,
                                  batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("Dataset does not exist")

    fps_index_all = []

    for data, label in data_loader:
        data, label = data.to(args.device), label.to(args.device).squeeze()
        fps_index = farthest_point_sample(data, args.num_regions)
        fps_index_all.append(fps_index.detach().cpu().numpy())

    fps_index_all = np.concatenate(fps_index_all) # should be (30,num_regions) ndarray
    print(fps_index_all)
    np.save('fps_%s_%d_%d_index_final30.npy'%(args.dataset, args.num_points, args.num_regions), fps_index_all)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save FPS index')
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N',choices=['modelnet10', 'shapenet'])
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--device_id', type=int, default=0, help='gpu id to use')  # change GPU here

    args = parser.parse_args()

    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)

    if args.cuda:
        print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        print('Using CPU')
    
    save_fps(args)
