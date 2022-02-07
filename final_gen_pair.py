import torch
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tools.final_util import set_random, set_model_args, mkdir, cal_rank, square_distance_np, load_model, ball_query, BALL_QUERY_COEF, get_folder_name_list
from tools.final_util import NUM_POINTS, NUM_REGIONS, SHAPENET_CLASS, SHAPENET_CAT2ID
from tools.final_common import get_reward
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test
from final_rotate_center_enum_all import rotate_xyz
from final_trans_center_enum_all import translate_pc
import itertools
from scipy.special import comb




def gen_context(region_pair_list, save_path, args):
    """ helper for save_context, generate context for a certain (i,j) pair list
    Input:
        args:
    Return:
        None
    """
    for ratio in args.ratio:
        context_list = []
        m = int((args.num_regions - 2) * ratio)  # m-order
        for pair_idx, region_pair in enumerate(region_pair_list):
            region_i, region_j = region_pair[0], region_pair[1]
            all_S = list(range(args.num_regions))
            all_S.remove(region_i)
            all_S.remove(region_j)

            if comb(len(all_S), m) > args.num_save_context_max:  # sample num_save_context_max contexts from N\{i,j}
                context_this_pair = []
                for k in range(args.num_save_context_max):
                    context_this_pair.append(np.random.choice(all_S, m, replace=False))
            else:  # enumerate all context S of length m in N\{i,j}, when m=0, this is empty list
                context_this_pair = list(itertools.combinations(all_S, m))
            context_list.append(context_this_pair)
        context_list = np.array(context_list)  # (num_pairs, num_context, m)  num_pairs is fixed by args, num_context is C_30^m or num_save_context_max
        print(context_list.shape)
        np.save(save_path + "ratio%d_context_list.npy" % (int(ratio * 100)),context_list)  # max pose and min pose will both use this context

def save_context(args):
    """ generate and save contexts for all randomly sampled pairs of regions and
    pairs containing the most rotation-sensitive region and all its neighbors
    Input:
        args:
    Return:
        None
    """
    print("gen context start...")
    for i, name in enumerate(folder_name_list):
        print("======= sample %s =======" % (name))
        base_folder = args.exp_folder + "%s/" % name
        interaction_folder = base_folder + "interaction_seed%d/" % args.seed
        single_region_folder = interaction_folder + "%s_adv_single_region/" % args.mode

        # gen context for interaction on global scale
        region_pair_list = np.load(interaction_folder + "region_pair_list.npy")
        gen_context(region_pair_list, interaction_folder, args)

        # gen context for interaction w.r.t a single region
        for region_folder_name in sorted(os.listdir(single_region_folder)):
            if not os.path.isdir(single_region_folder + region_folder_name): # not a directory, but a file
                continue
            print("----- %s ------" % (region_folder_name))
            region_folder = single_region_folder + region_folder_name + "/"
            region_pair_list = np.load(region_folder + "region_pair_list.npy")
            gen_context(region_pair_list, region_folder, args)



def gen_pred_label(model, data, lbl, disturb_fn, save_path, args):
    """ helper for save_pred_label """
    transform_params = np.load(save_path + "transform_params.npy").astype(np.float32)
    transform_params = torch.from_numpy(transform_params).to(data.device)
    data_disturb = disturb_fn(data, transform_params)
    if args.model == "pointnet":
        logits, _, _ = model(data_disturb.permute(0, 2, 1))
    else:
        logits = model(data_disturb.permute(0, 2, 1))
    pred = torch.argmax(logits, dim=1)
    with open(save_path + "pred_labels.txt", "w") as f:
        f.write("lbl: %d\npred_lbl: %d\n" % (lbl[0].item(), pred[0].item()))
    # print(np.array([lbl[0].item(), pred[0].item()]))
    np.save(save_path + "pred_labels.npy", np.array([lbl[0].item(), pred[0].item()]))

def save_pred_label(args, disturb_fn):
    """ save the prediction for each pose, along with the ground truth label """
    print("saving pred labels...")
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
            print("======= sample %s =========" % (name))
            data = data.to(args.device)
            lbl = lbl.to(args.device)
            base_folder = args.exp_folder + '%s/' % name
            interaction_folder = base_folder + "interaction_seed%d/" % args.seed
            single_region_folder = interaction_folder + "%s_adv_single_region/" % args.mode

            gen_pred_label(model, data ,lbl, disturb_fn, interaction_folder + "%s_adv/" % args.mode, args)

            for region_folder_name in sorted(os.listdir(single_region_folder)):
                if not os.path.isdir(single_region_folder + region_folder_name):
                    continue
                print("----- %s ------" % (region_folder_name))
                region_folder = single_region_folder + region_folder_name + "/"
                gen_pred_label(model, data, lbl, disturb_fn, region_folder + "max_pose/", args)
                gen_pred_label(model, data, lbl, disturb_fn, region_folder + "min_pose/", args)



def gen_pair_single_region(region, neighbor_idx, args):
    """ generate pairs containing the most rotation-sensitive region and all its neighbors (determined by ball query)
    Input:
        region: scalar, region to investigate (region i)
        neighbor_idx: (num_regions, num_regions) bool ndarray, neighbor index of all regions
    Return:
        region_pair_list: (num_neighbors, 2) ndarray, num_neighbors is not a fixed number
    """
    region_pair_list = []
    neighbors = np.arange(args.num_regions)[neighbor_idx[region]]  # this including region i itself
    for neighbor in neighbors:
        if region == neighbor:  # avoid (i,i) pair
            continue
        region_pair_list.append([region, neighbor])
    region_pair_list = np.array(region_pair_list)  # (num_neighbors, 2) num_neighbors is not a fixed number
    return region_pair_list


def save_pair_single_region(args):
    """ save the pairs containing the most rotation-sensitive region and all its neighbors
        The saved pairs will be used to compute the interaction of the most rotation-sensitive region
    Input:
        args:
    Return:
        None
    """
    print("gen pair single region...")
    assert args.mode == "trans" or args.mode == "rotate"
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
            data = data.cpu().numpy().squeeze() # (1024,3)
            print("====== sample %s =====" % (name))
            print(data.shape, np.abs(data).max())
            base_folder = args.exp_folder + "%s/" % name
            mode_folder = base_folder + '%s_all/' % args.mode
            interaction_folder = base_folder + "interaction_seed%d/" % args.seed
            single_region_folder = interaction_folder + "%s_adv_single_region/" % args.mode
            mkdir(single_region_folder)

            region_id = np.load(base_folder + "region_id.npy")  # (num_points,)
            region_shapley_values = np.load(mode_folder + "region_shapley_value.npy")  # (num_poses, num_regions)
            if args.mode == "trans":
                transform_params = np.load(mode_folder + "trans_vector.npy") # (num_poses,3)
            else: # args.mode == "rotate_all"
                transform_params = np.load(mode_folder + "angle_tuple.npy") # (num_poses,3)

            max_sv_per_region = np.max(region_shapley_values, axis=0) # (num_regions,) max_T phi_{x'=T(x)}(i)
            min_sv_per_region = np.min(region_shapley_values, axis=0) # (num_regions,) min_T phi_{x'=T(x)}(i)
            max_pose_idx = np.argmax(region_shapley_values, axis=0)  # (num_regions,)
            min_pose_idx = np.argmin(region_shapley_values, axis=0)  # (num_regions,)
            range_per_region = max_sv_per_region - min_sv_per_region # (num_regions,) range = max - min
            range_rank = args.num_regions - cal_rank(range_per_region) # (num_regions,) from 1 to num_regions, rank1 denotes the largest range

            # compute neighbors by ball query
            pairwise_distance = square_distance_np(data)  # (num_points, num_points)
            diameter = np.sqrt(np.maximum(pairwise_distance, 0)).max()
            region_centers = np.zeros((args.num_regions, 3))
            for i in range(args.num_regions):
                region_centers[i] = data[region_id == i].mean(axis=0)  # (3,)
            neighbor_idx = ball_query(region_centers, r=BALL_QUERY_COEF * diameter)  # (num_regions,num_regions) bool array
            # print("possible number of neighbor (i,j) pair: %d" % (np.sum(neighbor_idx) - args.num_regions)) # remove (i,i) pair

            for region in range(args.num_regions):
                region_folder = single_region_folder + "range_rank%02d_region%02d/" % (range_rank[region], region)
                # mkdir(region_folder)
                mkdir(region_folder + "normal/")
                mkdir(region_folder + "max_pose/")
                mkdir(region_folder + "min_pose/")
                max_pose_transform_params = transform_params[max_pose_idx[region]] # (3,)
                min_pose_transform_params = transform_params[min_pose_idx[region]] # (3,)
                np.save(region_folder + "max_pose/transform_params.npy", max_pose_transform_params) # transformation for pose of largest sv of this region
                np.save(region_folder + "max_pose/pose_idx.npy", max_pose_idx[region]) # scalar
                np.save(region_folder + "min_pose/transform_params.npy", min_pose_transform_params) # transformation for pose of smallest sv of this region
                np.save(region_folder + "min_pose/pose_idx.npy", min_pose_idx[region]) # scalar

                region_pair_list = gen_pair_single_region(region, neighbor_idx, args) # same pair list for both max and min pose
                print(region_pair_list.shape)
                if len(region_pair_list) == 0:
                    print("NO NEIGHBORS!!!")
                np.save(region_folder + "region_pair_list.npy", region_pair_list)


def check_adv_success(args, disturb_fn):
    """ check how many poses succesfully attack the model (i.e., the translated/rotated point cloud is misclassified)
        We will only choose point clouds that have at least one pose being misclassified, to compute the interaction of
        normal and adv samples.
        Also, this function will save the pose idx with the maximum attacking utility, i.e., the pose that has the lowest
        log p/(1-p) on the ground truth category. It will also save the corresponding transform parameters (in this case,
        the transform parameters are rotation angles) w.r.t. this pose.
    Input:
        args:
        disturb_fn: function to disturb the point cloud
    Return:
        None
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
            folder_name = folder_name_list[pc_idx]
            print("======= sample %s =========" % (folder_name))
            data = data.to(args.device)  # (B,num_points,3)
            lbl = lbl.to(args.device)  # (B,)

            base_folder = args.exp_folder + "%s/" % folder_name
            mode_folder = base_folder + '%s_all/' % args.mode
            interaction_folder = base_folder + "interaction_seed%d/" % args.seed

            if args.mode == "trans":
                all_transform_params = np.load(mode_folder + "trans_vector.npy")
            else:  # args.mode == "rotate":
                all_transform_params = np.load(mode_folder + "angle_tuple.npy")

            num_poses = all_transform_params.shape[0]
            all_data_disturb = []
            for i in range(num_poses):
                transform_params = torch.from_numpy(all_transform_params[i]).to(args.device)  # (3,) for translation and rotation
                data_disturb = disturb_fn(data, transform_params)  # (B,num_points,3) B=1, num_points=1024
                all_data_disturb.append(data_disturb)
            all_data_disturb = torch.cat(all_data_disturb, dim=0)  # (num_poses, num_points,3)

            if args.model == "pointnet":
                logits, _, _ = model(all_data_disturb.permute(0, 2, 1))
            else:
                logits = model(all_data_disturb.permute(0, 2, 1))  # (num_poses,num_class)
            pred = torch.argmax(logits, dim=1)  # (num_poses,)
            torch.cuda.empty_cache()

            num_miscls = (pred != lbl[0].item()).sum()
            print("%d poses are misclassified" % num_miscls)

            v = get_reward(logits, lbl, args)
            pose_idx_max_atk_util = torch.argmin(v).item()  # lowest log p/(1-p) on gt class
            transform_params_max_atk_util = all_transform_params[pose_idx_max_atk_util]
            # print(transform_params.shape)
            np.save(interaction_folder + "%s_adv/pose_idx.npy" % args.mode, pose_idx_max_atk_util)
            np.save(interaction_folder + "%s_adv/transform_params.npy" % args.mode, transform_params_max_atk_util)
            print("Pose idx with max attacking utility: %d" % pose_idx_max_atk_util)

def gen_pair_random(args):
    """ return randomly sampled pairs of regions
    Input:
        args:
    Return:
        region_pair_list: (num_pairs_random, 2) ndarray
    """
    all_pairs = [[i, j] for i in range(args.num_regions) for j in range(args.num_regions) if j > i]
    all_pairs = np.array(all_pairs)
    num_all_pairs = all_pairs.shape[0]
    pair_idx = np.random.choice(num_all_pairs, size=args.num_pairs_random, replace=False)
    region_pair_list = all_pairs[pair_idx]
    return region_pair_list

def save_pair_random(args):
    """ save the randomly samples pairs of regions
        The saved pairs will be used to compute the interaction of both the normal sample and the adv sample.
    Input:
        args:
    Return:
        None
    """
    print("gen pair random...")
    for i, name in enumerate(folder_name_list):
        print("====== sample %s =====" % (name))
        base_folder = args.exp_folder + "%s/" % name
        interaction_folder = base_folder + "interaction_seed%d/" % args.seed
        mkdir(interaction_folder)
        mkdir(interaction_folder + "normal/")
        mkdir(interaction_folder + "%s_adv/" % args.mode)
        region_pair_list = gen_pair_random(args)
        # print(region_pair_list.shape)
        np.save(interaction_folder + "region_pair_list.npy", region_pair_list) # for both normal pose and adv pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2', 'pointconv', 'dgcnn', 'gcnn', 'gcnn_adv'])
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N', choices=['modelnet10','shapenet'])
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--device_id', type=int, default=0)

    parser.add_argument("--mode", default='rotate', type=str)

    parser.add_argument("--ratio", default=[0., 0.04, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                        type=int)  # number of sample rate chosen from the uniform distribution in [0,1)
    parser.add_argument("--num_pairs_random", default=300, type=int)  # number of random pairs
    parser.add_argument("--num_save_context_max", default=100, type=int)  # max number of contexts for each I_ij
    parser.add_argument('--softmax_type', default='modified', type=str, choices=["normal","modified"])
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

    # sample pairs for computing the average interaction over all regions
    save_pair_random(args)

    # check if one of the 216 poses successfully generate an adv sample
    if args.mode == "trans":
        check_adv_success(args, disturb_fn=translate_pc)
    else: # args.mode == "rotate"
        check_adv_success(args, disturb_fn=rotate_xyz)

    # sample pairs for the most ratation-sensitive region
    save_pair_single_region(args)
    # save context S for random pairs and pairs related to the most ratation-sensitive region
    save_context(args)

    if args.mode == "trans":
        save_pred_label(args, disturb_fn=translate_pc)
    else: # args.mode == "rotate"
        save_pred_label(args, disturb_fn=rotate_xyz)


