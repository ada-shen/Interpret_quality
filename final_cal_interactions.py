import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader

from tools.final_util import set_random, set_model_args, get_folder_name_list
from tools.final_util import NUM_REGIONS, NUM_POINTS, MODELNET_INTER_SELECTED_SAMPLE, SHAPENET_INTER_SELECTED_SAMPLE, SHAPENET_CLASS
from tools.final_common import get_reward
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test



def compute_order_interaction(all_logits, lbl, args):
    ''' compute the interaction between two regions (their indices are specified in region pair) in one single
        pointcloud, with the given contexts all_S.
        Input:
            all_logits: (num_pairs, 4 * num_context, num_class) tensor
            lbl: (B,) tensor, B=1
        Return:
            all_interaction: (num_pairs, num_context) ndarray
    '''
    num_pairs = all_logits.size()[0]
    num_context = all_logits.size()[1] // 4
    all_interaction = []

    with torch.no_grad():
        for i in range(num_pairs):
            logits = all_logits[i, :, :] # (4 * num_context, num_class) tensor
            v = get_reward(logits, lbl, args)
            # v is now of shape (4 * num_context,)
            for k in range(num_context):
                score = v[4 * k] + v[4 * k + 3] - v[4 * k + 1] - v[4 * k + 2]
                all_interaction.append(score.item())

        all_interaction = np.array(all_interaction).reshape(-1, num_context) # (num_pairs, num_context)
        return all_interaction


def cal_interaction_all_orders(lbl, save_path, args):
    for ratio in args.ratio:
        print('\tratio: %f' % (ratio))
        all_logits = torch.load(save_path + "ratio%d_all_logits.pt" % (int(ratio * 100)))  # (num_pairs, 4 * num_context, num_class) tensor
        all_interaction = compute_order_interaction(all_logits, lbl, args)
        print(all_interaction.shape)
        np.save(save_path + "ratio%d_%s_interaction.npy" % (int(ratio * 100), args.output_type), all_interaction)


def cal_interaction(args):
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

            # normal pose
            print("##### normal pose")
            cal_interaction_all_orders(lbl, interaction_folder + "normal/", args)

            # adv pose with max attacking utility
            print("##### max attacking utility pose")
            pred_class = np.load(interaction_folder + "%s_adv/pred_labels.npy" % args.mode)[1]
            print("pred: ", pred_class)
            pred = torch.tensor([pred_class], dtype=torch.long, device=args.device)
            if args.output_type == "gt":
                cal_interaction_all_orders(lbl, interaction_folder + "%s_adv/" % args.mode, args)
            else:
                print("using pred, ", pred)
                cal_interaction_all_orders(pred, interaction_folder + "%s_adv/" % args.mode, args)

            for region_folder_name in sorted(os.listdir(single_region_folder)): # interaction for single regions
                if not os.path.isdir(single_region_folder + region_folder_name):
                    continue
                print("----- %s ------" % (region_folder_name))
                range_rank = int(region_folder_name[10:12])  # get range rank information from folder name, 1-based rank
                if range_rank != 1: # we only compute the interaction of the *most* rotation-sensitive region at *normal* pose
                    continue
                region_folder = single_region_folder + region_folder_name + "/"

                # interaction of abnormal region (most rotation-sentitive region) at normal pose
                cal_interaction_all_orders(lbl, region_folder + "normal/", args) # normal pose is always correctly classified, so lbl==pred, and using either one is ok



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2', 'pointconv', 'dgcnn', 'gcnn', 'gcnn_adv'])
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N',
                        choices=['modelnet10', 'shapenet'])
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--gen_pair_seed', type=int, default=1, help='seed used in gen_pair, only used for checking instability')
    parser.add_argument('--device_id', type=int, default=1)

    parser.add_argument("--mode", default='rotate', type=str)

    parser.add_argument("--ratio", default=[0., 0.04, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                        type=int)  # number of sample rate chosen from the uniform distribution in [0,1)
    parser.add_argument('--softmax_type', default='modified', type=str, choices=["normal","modified","yi","minuslog"])
    parser.add_argument('--output_type', default='pred', type=str, choices=["gt", "pred"])
    parser.add_argument("--num_pairs_random", default=300,type=int)  # number of random pairs when gen_pair_type is random
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

    folder_name_list = get_folder_name_list(args)
    if args.dataset == "modelnet10":
        selected_sample_idx = MODELNET_INTER_SELECTED_SAMPLE
    else:
        selected_sample_idx = SHAPENET_INTER_SELECTED_SAMPLE

    cal_interaction(args)





