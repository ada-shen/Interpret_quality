import os
import argparse
import torch
import numpy as np
import math
from tools.final_util import set_random, set_model_args, set_shapley_batch_size
from tools.final_util import NUM_POINTS, NUM_REGIONS, NUM_SAMPLES
from tools.final_common import test

MODE = "rotate"
ANGLE_THRESHOLD = math.pi / 4
NUM_GRID_ENUM_ROTATE = 6


def rotate_xyz(x, angle_tuple):
    """ rotate the point cloud by x, y, z angles
    Input:
        x: (B,num_points,3) tensor, B=1, point cloud
        angle_tuple: (3,) tensor, i.e. (theta_x, theta_y, theta_z)
    Return:
        (B,num_points,3) tensor, rotated pointcloud
    """
    B = x.shape[0]
    theta_x, theta_y, theta_z = angle_tuple[0], angle_tuple[1], angle_tuple[2]
    cos_x, cos_y, cos_z = torch.cos(theta_x), torch.cos(theta_y), torch.cos(theta_z)
    sin_x, sin_y, sin_z = torch.sin(theta_x), torch.sin(theta_y), torch.sin(theta_z)
    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                  [0, cos_x, -sin_x],
                                  [0, sin_x, cos_x]], device=x.device)
    rotation_matrix_y = torch.tensor([[cos_y, 0, sin_y],
                                  [0, 1, 0],
                                  [-sin_y, 0, cos_y]], device=x.device)
    rotation_matrix_z = torch.tensor([[cos_z, -sin_z, 0],
                                  [sin_z, cos_z, 0],
                                  [0, 0, 1]], device=x.device)
    rotation_matrix = torch.matmul(torch.matmul(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    x_rotate = torch.matmul(x, rotation_matrix.expand(B, 3, 3).permute(0, 2, 1)) # (R x.T).T = x R.T
    return x_rotate


def generate_rotate_angle(args, device):
    """ returns sampled rotation angle tuples
    Return: (num_grid_enum_rotate^3, 3) tensor
    """
    theta_x_all = np.linspace(-args.angle_threshold, args.angle_threshold, num=args.num_grid_enum_rotate)
    theta_y_all = np.linspace(-args.angle_threshold, args.angle_threshold, num=args.num_grid_enum_rotate)
    theta_z_all = np.linspace(-args.angle_threshold, args.angle_threshold, num=args.num_grid_enum_rotate)
    Theta_x, Theta_y, Theta_z = np.meshgrid(theta_x_all, theta_y_all, theta_z_all, indexing='ij')
    all_rotate_angle = []
    for i in range(args.num_grid_enum_rotate):
        for j in range(args.num_grid_enum_rotate):
            for k in range(args.num_grid_enum_rotate):
                theta_tuple = torch.tensor([Theta_x[i, j, k],Theta_y[i, j, k],Theta_z[i, j, k]], dtype=torch.float32)
                all_rotate_angle.append(theta_tuple)

    all_rotate_angle = torch.stack(all_rotate_angle, dim=0)  # (num_grid_enum_rotate^3, 3)
    all_rotate_angle = all_rotate_angle.to(device)
    return all_rotate_angle


def print_rotate_info(io, angle_tuple, region_shapley_value, epoch):
    io.cprint("rotation angle: [%f pi, %f pi, %f pi]" % (
        angle_tuple[0].item() / np.pi, angle_tuple[1].item() / np.pi, angle_tuple[2].item() / np.pi))
    io.cprint("shapley value after %d epoch:\n%s" % (epoch, str(region_shapley_value)))


def save_rotate_info(all_rotate_angle, result_path):
    np.save(result_path + "angle_tuple.npy", all_rotate_angle.cpu().numpy())

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
    args.angle_threshold = ANGLE_THRESHOLD
    args.num_grid_enum_rotate = NUM_GRID_ENUM_ROTATE
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)
    set_model_args(args)
    set_shapley_batch_size(args)  # set different batch size for different models

    test(args, get_transform_params_fn=generate_rotate_angle, disturb_fn=rotate_xyz,
         print_info_fn=print_rotate_info, save_info_fn=save_rotate_info)

if __name__ == "__main__":
    main()