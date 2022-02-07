import os
import argparse
import torch
import numpy as np
from tools.final_util import set_model_args, set_random, set_shapley_batch_size
from tools.final_util import NUM_POINTS, NUM_REGIONS, NUM_SAMPLES
from tools.final_common import test

MODE = "trans"
TRANS_DIST_THRESHOLD = 0.5
NUM_GRID_ENUM_TRANS = 6

def translate_pc(data, trans):
    """ translate the point cloud
    Input:
        data: (B,num_points,3) tensor
        trans: (3,) tensor
    Return:
        translated point cloud, (B,num_points,3) tensor
    """
    return torch.add(data, trans)


def generate_trans_vector(args, device):
    """ generate a sequence of translation vectors, according to number of samples per axis and distance bound
    Return: (num_grid_enum_trans^3, 3) tensor, containing all possible translation vectors
    """
    x = np.linspace(-args.trans_dist_threshold, args.trans_dist_threshold, num=args.num_grid_enum_trans)
    y = np.linspace(-args.trans_dist_threshold, args.trans_dist_threshold, num=args.num_grid_enum_trans)
    z = np.linspace(-args.trans_dist_threshold, args.trans_dist_threshold, num=args.num_grid_enum_trans)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    all_trans_vector = []
    for i in range(args.num_grid_enum_trans):
        for j in range(args.num_grid_enum_trans):
            for k in range(args.num_grid_enum_trans):
                trans = torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]], dtype=torch.float32)
                if torch.norm(trans) > args.trans_dist_threshold:
                    trans = trans / torch.norm(trans) * args.trans_dist_threshold
                all_trans_vector.append(trans)

    all_trans_vector = torch.stack(all_trans_vector, dim=0) # (num_grid_enum_trans^3, 3)
    all_trans_vector = all_trans_vector.to(device)
    return all_trans_vector


def print_trans_info(io, trans, region_shapley_value, epoch):
    io.cprint("translation vector: [%f, %f, %f]" % (trans[0].item(), trans[1].item(), trans[2].item()))
    io.cprint("translation distance: %f" % torch.norm(trans).item())
    io.cprint("shapley value after %d epoch:\n%s" % (epoch, str(region_shapley_value)))


def save_trans_info(all_trans_vector, result_path):
    np.save(result_path + "trans_vector.npy", all_trans_vector.cpu().numpy())
    np.save(result_path + "trans_distance.npy", torch.norm(all_trans_vector, dim=1).cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='gcnn_adv', metavar='N',
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
    args.trans_dist_threshold = TRANS_DIST_THRESHOLD
    args.num_grid_enum_trans = NUM_GRID_ENUM_TRANS
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)
    set_model_args(args)
    set_shapley_batch_size(args) # set different batch size for different models

    test(args, get_transform_params_fn=generate_trans_vector, disturb_fn=translate_pc,
         print_info_fn=print_trans_info, save_info_fn=save_trans_info)

if __name__ == "__main__":
    main()