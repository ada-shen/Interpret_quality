""" save the region id and sampled orders
also calculate the Shapley value for each region (at the original position of the point cloud) """
import os

import argparse
import torch
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test
import numpy as np
from torch.utils.data import DataLoader
from tools.final_util import IOStream,set_random, set_model_args, load_model, get_folder_name_list, square_distance, mkdir
from tools.final_util import NUM_POINTS, NUM_REGIONS, NUM_SAMPLES_SAVE, SHAPENET_CLASS   # constants
from tools.final_common import cal_reward

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(args.exp_folder):
        os.makedirs(args.exp_folder)

def cal_region_id(data, fps_index, result_path, save=True):
    """ calculate and save region id of all the points
    Input:
        data: (B,num_points,3) tensor, point cloud, num_points=1024
        fps_index: (num_regions,) ndarray, center idx of the 32 regions
        result_path: path to save file
    Return:
        region_id: (num_points,) ndarray, record each point belongs to which region
    """
    data_fps = data[:, fps_index, :]  # (B, num_regions, 3), centroids of each region
    distance = square_distance(data, data_fps)  # (B, num_points, num_regions), here B=1
    region_id = torch.argmin(distance, dim=2)  # (B, num_points), B=1
    region_id = region_id.squeeze().cpu().numpy() # (num_points,) ndarray
    if save:
        np.save(result_path + "region_id.npy", region_id)  # (num_points,)
    return region_id



def cal_norm_factor(model, data, lbl, center, result_path, args, save=True):
    """ calculate v(N) - v(empty)
    Input:
        data: (B,num_points,3) tensor, point cloud (already transposed)
        lbl: (B,) tensor, label
        center: (3,) tensor, center of point cloud
        result_path: path to save
    Return:
        norm_factor: scalar, v(Omega) - v(empty)
    """
    B = data.shape[0]
    empty = center.view(1, 1, 3).expand(B, args.num_points, 3).clone()
    v_N, _ = cal_reward(model, data, lbl, args)
    v_empty, _ = cal_reward(model, empty, lbl, args)
    norm_factor = (v_N - v_empty).cpu().item()
    if save:
        np.save(result_path + "norm_factor.npy", norm_factor)
    return norm_factor


def generate_all_orders(result_path, args, save=True):
    """ generate random orders for sampling
    Input:
        result_path: path to save all orders
    Return:
        all_orders: (num_samples_save, num_regions) ndarray
    """
    all_orders = []
    for k in range(args.num_samples_save):
        all_orders.append(np.random.permutation(np.arange(0, args.num_regions, 1)).reshape((1, -1))) # append (1,num_regions)
    all_orders = np.concatenate(all_orders, axis=0)  # (num_samples_save, num_regions)
    if save:
        np.save(result_path + "all_orders.npy", all_orders)
    return all_orders

def mask_data(masked_data, center, order, region_id):
    """ mask the data to the center of the point cloud
    Input:
        masked_data: (region+1, num_points,3) tensor, data to be masked
        center: (3,) tensor, center of point cloud
        order: (num_regions,) ndarray
        region_id: (num_points,) ndarray
    Return:
        masked_data: (region+1, num_points,3) tensor, modified
    """
    for j in range(1, len(order) + 1):
        mask_region_id = order[j - 1]
        mask_index = (region_id == mask_region_id)
        masked_data[:j, mask_index, :] = center
    return masked_data


def save_shapley(region_shap_value, pc_idx, count, result_path, region_id, args):
    N = args.num_points
    shap_value = np.zeros((N,))

    folder = result_path + "shapley/"
    mkdir(folder)

    folder2 = result_path + "region_shapley/"
    mkdir(folder2)

    for k in range(0, args.num_regions):
        region_index = (region_id == k)
        shap_value[region_index] = region_shap_value[k] / count

    np.save(folder + "%s.npy" % (str(pc_idx) + '_' + str(count)), shap_value)
    np.save(folder2 + "%s.npy" % (str(pc_idx) + '_' + str(count)), region_shap_value / count)



def shap_sampling(model, dataloader, args, folder_name_list):
    sample_nums = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

    with torch.no_grad():
        model.eval()
        fps_indices = np.load('fps_%s_%d_%d_index_final30.npy'%(args.dataset, args.num_points, args.num_regions))
        fps_indices = torch.from_numpy(fps_indices).to(args.device)

        for i, (data, lbl) in enumerate(dataloader):
            B, N = data.shape[0], args.num_points
            folder_name = folder_name_list[i]
            result_path = args.exp_folder + '%s/' % folder_name
            mkdir(result_path)

            count = 0
            region_sv_all = []  # (num_samples_save, num_regions)
            region_shap_value = np.zeros((args.num_regions,)) # (num_regions,)

            data = data.to(args.device) # (B, num_points, 3), here B=1
            lbl = lbl.to(args.device) # (B,), here B=1
            fps_index = fps_indices[i] # (num_regions,)
            region_id = cal_region_id(data, fps_index, result_path, save=True) # (num_points,)

            center = torch.mean(data, dim=1).squeeze() # (3,)

            norm_factor = cal_norm_factor(model, data, lbl, center, result_path, args, save=True)
            all_orders = generate_all_orders(result_path, args, save=True) # (num_samples_save, num_regions) int array

            while count < args.num_samples_save:
                print("pointcloud:%s, index:%d, sample:%d" % (folder_name, i, count))
                order = all_orders[count]  # Sample an order
                masked_data = data.expand(args.num_regions + 1, N, 3).clone()
                masked_data = mask_data(masked_data, center, order, region_id)

                v, _ = cal_reward(model, masked_data, lbl, args) # (num_regions+1,) tensor
                dv = v[1:] - v[:-1]
                region_shap_value[order] += (dv.cpu().numpy())

                temp = np.zeros((args.num_regions,))
                temp[order] += dv.cpu().numpy()
                region_sv_all.append(temp)
                count += 1

                if count in sample_nums:
                    save_shapley(region_shap_value, i, count, result_path, region_id, args)

            np.save(result_path + "region_sv_all.npy", region_sv_all) # (num_samples_save, num_regions)


def test(args):

    if args.dataset == "modelnet10":
        data_loader = DataLoader(ModelNet_Loader_Shapley_test(args, partition='train', num_points=args.num_points),
                                 num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "shapenet":
        data_loader = DataLoader(ShapeNetDataset_Shapley_test(args, split='train', npoints=args.num_points,
                                                              class_choice=SHAPENET_CLASS, classification=True),
                                 num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("Dataset does not exist")
    model = load_model(args)
    folder_name_list = get_folder_name_list(args)
    shap_sampling(model, data_loader, args, folder_name_list)


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointconv', metavar='N',
                        choices=['pointnet', 'dgcnn', 'gcnn', 'pointnet2', 'pointconv', 'gcnn_adv'])
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N',
                        choices=['modelnet10', 'shapenet'])
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--device_id', type=int, default=0, help='gpu id to use')  # change GPU here
    parser.add_argument('--softmax_type', type=str, default="modified", choices=["normal", "modified"])

    args = parser.parse_args()

    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    args.num_samples_save = NUM_SAMPLES_SAVE
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    _init_(args)

    set_random(args.seed)
    set_model_args(args)

    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        print('Using CPU')

    test(args)


if __name__ == "__main__":
    main()
