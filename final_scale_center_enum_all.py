import os
import argparse
import torch
import numpy as np
from tools.final_util import set_random, set_shapley_batch_size, set_model_args
from tools.final_util import NUM_POINTS, NUM_REGIONS, NUM_SAMPLES
from tools.final_common import test

MODE = "scale"
SCALE_UPPER = 2.0
SCALE_LOWER = 0.5
NUM_GRID_ENUM_SCALE = 30

def scale_pc(data, scale):
    """ scale the point cloud
    Input:
        data: (B,num_points,3) tensor
        scale: scalar tensor
    Return:
        (B,num_points,3) tensor, scaled point cloud
    """
    return data * scale


def generate_scale(args, device):
    """ generate a sequence of scales, according to number of samples in args
    Return: (num_grid_enum_scale,) tensor, containing all possible scales
    """
    all_scale = np.linspace(start=args.scale_lower, stop=args.scale_upper, num=args.num_grid_enum_scale) #(num_grid_enum_scale,)
    all_scale = torch.from_numpy(all_scale).float().to(device)
    return all_scale


def print_scale_info(io, scale, region_shapley_value, epoch):
    io.cprint("scale: %f" % scale)
    io.cprint("shapley value after %d epoch:\n%s" % (epoch, str(region_shapley_value)))


def save_scale_info(all_scale, result_path):
    np.save(result_path + "scale.npy", all_scale.cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointconv', metavar='N',
                        choices=['pointnet', 'dgcnn', 'gcnn', 'pointnet2', 'pointconv', 'gcnn_adv'])
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N', choices=['modelnet10','shapenet'])
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--device_id', type=int, default=0, help='gpu id to use')  # change GPU here
    parser.add_argument('--softmax_type', type=str, default="modified", choices=["normal", "modified"])

    args = parser.parse_args()

    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    args.num_samples = NUM_SAMPLES
    args.mode = MODE
    args.scale_upper = SCALE_UPPER
    args.scale_lower = SCALE_LOWER
    args.num_grid_enum_scale = NUM_GRID_ENUM_SCALE
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)
    set_model_args(args)
    set_shapley_batch_size(args) # set different batch size for different models

    test(args, get_transform_params_fn=generate_scale, disturb_fn=scale_pc,
         print_info_fn=print_scale_info, save_info_fn=save_scale_info)

if __name__ == "__main__":
    main()