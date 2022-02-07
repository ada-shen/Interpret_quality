import torch
import os
import numpy as np
import argparse
import time
import math
from torch.utils.data import DataLoader
from tools.final_util import set_random, set_model_args, load_model, mkdir, get_folder_name_list, set_interaction_batch_size
from tools.final_util import NUM_REGIONS, NUM_POINTS, SHAPENET_CLASS, MODELNET_INTER_SELECTED_SAMPLE, SHAPENET_INTER_SELECTED_SAMPLE
from final_rotate_center_enum_all import rotate_xyz
from final_trans_center_enum_all import translate_pc
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test


def compute_order_interaction_logits(model, data_disturb, region_id, region_pair_list, context_list, args):
    ''' compute the interaction between two regions (their indices are specified in region pair) in one single
        pointcloud, with the given contexts all_S.
        Input:
            data_disturb: (B,num_points,3) tensor, B=1, num_points=1024, disturbed pointcloud
            region_id: (num_points,) ndarray, num_points=1024
            region_pair_list: (num_pairs, 2) ndarray
            context_list: (num_pairs, num_context, m) ndarray, m is the order
        Return:
            all_interaction: (num_pairs, num_context) ndarray
    '''
    t0 = time.time()
    N = data_disturb.size()[1]
    num_context = context_list.shape[1]
    bs = args.interaction_batch_size
    iteration = math.ceil(num_context / bs)

    center = torch.mean(data_disturb, dim=1).squeeze()  # (3,) tensor
    data_disturb_permute = data_disturb.permute(0,2,1) # (B,3,N) tensor, B=1, N=1024
    all_logits = []

    with torch.no_grad():
        for pair_idx, region_pair in enumerate(region_pair_list):
            region_i, region_j = region_pair[0], region_pair[1]
            context_this_pair = context_list[pair_idx]  # (num_context, m)
            logits_this_pair = []
            for i in range(iteration):
                context_batch = context_this_pair[i * bs : min(num_context, (i + 1) * bs)] # (bs, m) or (num_context-i*bs, m)
                bs_use = context_batch.shape[0] # bs or num_context-i*bs (last iteration)
                # print("bs use=", bs_use)
                data_expand = data_disturb_permute.expand(4 * bs_use, -1, -1).clone() # (4*bs_use,3,N)
                mask = torch.zeros_like(data_expand)  # (4*bs_use,3,N) mask consists of 0 and 1
                for k in range(bs_use):
                    mask[4 * k:4 * (k + 1), :, np.in1d(region_id, context_batch[k])] = 1  # S
                    mask[4 * k + 1, :, region_id == region_i] = 1  # S U {i}
                    mask[4 * k + 2, :, region_id == region_j] = 1  # S U {j}
                    mask[4 * k, :, region_id == region_i] = 1
                    mask[4 * k, :, region_id == region_j] = 1  # S U {i,j}
                mask_zero2center = center.view(1,3,1).expand(4 * bs_use, -1, N).clone() # (4*bs_use,3,N) turn entries of value 0 to center
                mask_zero2center[mask == 1] = 0  # it has value of 0 where mask is 1, and value of center where mask is 0
                masked_data = data_expand * mask
                masked_data += mask_zero2center # turn 0 entries to center
                if args.model == "pointnet":
                    logits,_,_ = model(masked_data) # (4*bs_use, num_class) tensor
                else:
                    logits = model(masked_data)  # (4*bs_use, num_class) tensor
                # print("shape of logits: ", logits.size())
                logits_this_pair.append(logits.detach())
            logits_this_pair = torch.cat(logits_this_pair, dim=0) # (4 * num_context, num_class) tensor
            # print("shape of logits_this_pair: ", logits_this_pair.size())
            all_logits.append(logits_this_pair.unsqueeze(0)) # append (1, 4 * num_context, num_class) tensor
        all_logits = torch.cat(all_logits, dim=0) # (num_pairs, 4 * num_context, num_class) tensor
        print("shape of all_logits: ", all_logits.size())
        t1 = time.time()
        print("done time: ", t1-t0)
        return all_logits


def save_logits_all_orders(model, data, region_id, save_path, args):
    """ data can be normal pose, or adv pose"""
    region_pair_list = np.load(save_path + "../region_pair_list.npy")  # (num_pairs, 2)
    for ratio in args.ratio:
        print('\tratio: %f' % (ratio))
        context_list = np.load(save_path + "../ratio%d_context_list.npy" % (int(ratio * 100)))  # (num_pairs, num_context, m), num_context at most 100, m can be 0
        all_logits = compute_order_interaction_logits(model, data, region_id,region_pair_list, context_list, args)
        torch.save(all_logits, save_path + "ratio%d_all_logits.pt" % (int(ratio * 100)))


def save_logits(args, disturb_fn):
    """
    :param disturb_fn: function,   apply disturbance to the point cloud, choice = [translate_pc, rotate_xyz]
    """
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

    with torch.no_grad():
        for pc_idx, (data, lbl) in enumerate(data_loader):
            name = folder_name_list[pc_idx]
            if pc_idx not in selected_sample_idx:
                continue
            print("======= sample %s =========" % (name))
            data = data.to(args.device)
            lbl = lbl.to(args.device)

            base_folder = args.exp_folder + '%s/' % name
            interaction_folder = base_folder + "interaction_seed%d/" % args.gen_pair_seed
            single_region_folder = interaction_folder + "%s_adv_single_region/" % args.mode

            region_id = np.load(base_folder + "region_id.npy")

            # compute logits for normal pose
            save_logits_all_orders(model, data, region_id, interaction_folder + "normal/", args)

            # compute logits for adv pose with max attacking utility
            transform_params = np.load(interaction_folder + "%s_adv/transform_params.npy"%args.mode).astype(np.float32)
            transform_params = torch.from_numpy(transform_params).to(args.device)
            data_disturb = disturb_fn(data, transform_params)
            save_logits_all_orders(model, data_disturb, region_id, interaction_folder + "%s_adv/" % args.mode, args)

            # compute logits for single regions
            for region_folder_name in sorted(os.listdir(single_region_folder)):
                if not os.path.isdir(single_region_folder + region_folder_name):
                    continue
                print("----- %s ------" % (region_folder_name))
                range_rank = int(region_folder_name[10:12]) # get range rank information from folder name, 1-based rank
                if range_rank != 1: # we only compute the interaction of the *most* rotation-sensitive region at *normal* pose
                    continue
                region_folder = single_region_folder + region_folder_name + "/"

                # interaction of abnormal region (most rotation-sentitive region) at normal pose
                save_logits_all_orders(model, data, region_id, region_folder + "normal/", args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2', 'pointconv', 'dgcnn', 'gcnn', 'gcnn_adv'])
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N', choices=['modelnet10', 'shapenet'])
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--gen_pair_seed', type=int, default=1, help='seed used in gen_pair.py, only used for checking instability')
    parser.add_argument('--device_id', type=int, default=0)

    parser.add_argument("--mode", default='rotate', type=str)

    parser.add_argument("--ratio", default=[0., 0.04, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                        type=int)  # number of sample rate chosen from the uniform distribution in [0,1)
    parser.add_argument("--num_pairs_random", default=300, type=int)  # number of random pairs when gen_pair_type is random
    parser.add_argument("--num_save_context_max", default=100, type=int)  # # max number of contexts for each I_ij
    args = parser.parse_args()

    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)
    set_model_args(args)
    set_interaction_batch_size(args)

    folder_name_list = get_folder_name_list(args)
    if args.dataset == "modelnet10":
        selected_sample_idx = MODELNET_INTER_SELECTED_SAMPLE
    else:
        selected_sample_idx = SHAPENET_INTER_SELECTED_SAMPLE

    if args.mode == "trans":
        save_logits(args, disturb_fn=translate_pc)
    else: # args.mode == "rotate"
        save_logits(args, disturb_fn=rotate_xyz)





