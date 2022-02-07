import torch
import torch.nn as nn
import numpy as np
import time
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tools.final_util import IOStream, mkdir, load_model, get_folder_name_list, SHAPENET_CLASS


def get_reward(logits, lbl, args):
    """ given logits, calculate score for Shapley value or interaction
    Input:
        logits: (B', num_class) tensor, B' can be (num_region+1)*bs or B'=1
        lbl: (B,) tensor, B=1, label
    Return:
        v: (B',) tensor, reward score
    """
    num_class = logits.size()[1]
    if args.softmax_type == "normal":
        v = F.log_softmax(logits, dim=1)[:, lbl[0]]
    else: # args.softmax_type == "modified":
        v = logits[:, lbl[0]] - torch.logsumexp(logits[:, np.arange(num_class) != lbl[0].item()], dim=1)
    return v

def cal_reward(model, data, lbl, args):
    """ given data and label, calculate score for Shapley value or interaction
    Input:
        data: (B',num_points,3) tensor, B' can be (num_region+1)*bs or B=1, num_points=1024
        lbl: (B,) tensor, B=1, label
    Return:
        v: (B',) tensor, reward score
        logits: (B', num_class) tensor, logits for saving
    """
    data = data.permute(0,2,1).contiguous() #(B',3,num_points)
    if args.model == "pointnet" or args.model == "pointnet_roty_da":
        logits, _, _ = model(data) # (B',num_class)
    else:
        logits = model(data)

    v = get_reward(logits, lbl, args)

    return v, logits


def mask_data_batch(masked_data, center, orders, region_id, args):
    """ mask the point cloud to center by region, implemented in batch
    Input:
        masked_data: ((num_regions + 1) * bs, num_points, 3) tensor, data to be masked
        center: (3,) tensor, center of the point cloud
        orders: (bs,num_regions) ndarray, a batch of orders for masking
        region_id: (num_points,) ndarray, record each point belongs to which region
    Return:
        masked_data: ((num_regions + 1) * bs, num_points, 3) tensor, modified
    """
    for o_idx, order in enumerate(orders): # for each order/permutation in the batch
        for j in range(1, len(order) + 1):
            mask_region_id = order[j - 1]
            mask_index = (region_id == mask_region_id)
            masked_data[(args.num_regions + 1) * o_idx:(args.num_regions + 1) * o_idx + j, mask_index, :] = center
    return masked_data


def shap_sampling_all_regions_batch(model, data_disturb, lbl, region_id, load_order_list, args):
    """ calculate shapley value for all regions on the disturbed point cloud
    Input:
        data_disturb: (B,num_points,3) tensor, B=1, num_points=1024, disturbed point cloud
        lbl: (B,) tensor, B=1
        region_id: (num_points,) ndarray, record that each point belongs to which region
        load_order_list: (num_samples_save, num_regions) ndarray, all orders saved previously
    Return:
        region_shap_value: (num_regions,) ndarray, shapley value of all regions
        all_logits_this_pose: (num_samples * (num_regions+1), num_class) tensor, saved logits
    """
    N = args.num_points
    num_regions = args.num_regions
    bs = args.shapley_batch_size
    iterations = args.num_samples // bs # number of batches we need to iterate through

    center = torch.mean(data_disturb, dim=1).squeeze()  # (3,) tensor

    with torch.no_grad():
        region_shap_value = np.zeros((num_regions,))
        all_logits_this_pose = [] # we also save the logits in case we need it for other use
        t_start = time.time()
        for i in range(iterations):
            orders = load_order_list[i * bs: (i + 1) * bs]
            masked_data = data_disturb.expand((num_regions + 1) * bs, N, 3).clone()
            masked_data = mask_data_batch(masked_data, center, orders, region_id, args)

            v, logits = cal_reward(model, masked_data, lbl, args) # v is ((num_regions + 1) * bs,), logits is ((num_regions + 1) * bs, num_class)
            all_logits_this_pose.append(logits)
            for o_idx, order in enumerate(orders):
                v_single_order = v[(num_regions + 1) * o_idx: (num_regions + 1) * (o_idx + 1)]  # (num_regions+1,)
                dv = v_single_order[1:] - v_single_order[:-1]
                region_shap_value[order] += (dv.cpu().numpy())
        region_shap_value /= args.num_samples
        all_logits_this_pose = torch.cat(all_logits_this_pose, dim=0) # (num_samples * (num_regions+1), num_class) tensor
        assert all_logits_this_pose.size()[0] == args.num_samples * (num_regions+1)

    t_end = time.time()
    print("done time: ", t_end-t_start)
    return region_shap_value, all_logits_this_pose



def test(args, get_transform_params_fn, disturb_fn, print_info_fn, save_info_fn):
    """ program runner
    Input:
        args: parsed arguments
        get_transform_params_fn: function,   for translation: get all translation vectors,
            for rotation: get all rotate angle tuples, for scale: get all scales
            choice = [generate_trans_vector, generate_rotate_angle, generate_scale]
        disturb_fn: function,   apply disturbance to the point cloud
            choice = [translate_pc, rotate_xyz, scale_pc]
        print_info_fn: function,   print information
        save_info_fn: function,   save information
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

    folder_name_list = get_folder_name_list(args)

    for pc_index, (data, lbl) in enumerate(data_loader):
        data = data.to(args.device) # (B, num_points, 3), here B=1, num_points=1024
        lbl = lbl.to(args.device) # (B,), here B=1

        folder_name = folder_name_list[pc_index]
        base_folder = args.exp_folder + "%s/" % folder_name
        mode_folder = base_folder + "%s_all/" % args.mode
        mkdir(mode_folder)
        io = IOStream(mode_folder + "log.txt")
        io.cprint(str(args))

        norm_factor = np.load(base_folder + "norm_factor.npy")
        io.cprint("norm factor: %f" % norm_factor)
        region_id = np.load(base_folder + "region_id.npy") # (num_points,)
        load_order_list = np.load(base_folder + "all_orders.npy") # (num_samples_save,num_regions)

        t_start = time.time()
        region_shapley_list = []
        all_logits_list = []
        orig_region_shap_value, _ = shap_sampling_all_regions_batch(model, data, lbl, region_id, load_order_list, args)
        io.cprint("origin region shapley: %s" % str(orig_region_shap_value))
        np.save(mode_folder + "orig_shapley_value.npy", orig_region_shap_value)

        all_transform_params = get_transform_params_fn(args, data.device)
        for i in range(all_transform_params.size()[0]):
            transform_param = all_transform_params[i] # (3,) for translation and rotation, scalar for scale
            data_disturb = disturb_fn(data, transform_param) # (B,num_points,3) B=1
            region_shap_value, all_logits_this_pose = shap_sampling_all_regions_batch(model, data_disturb, lbl, region_id, load_order_list, args)
            # region_shap_value: (num_regions,) array, all_logits_this_pose: (num_samples * (num_regions+1), num_class) tensor
            region_shapley_list.append(region_shap_value)
            all_logits_list.append(all_logits_this_pose)
            print_info_fn(io, transform_param, region_shap_value, i)

        region_shapley_list = np.array(region_shapley_list)  # (num_poses, num_regions) array, num_poses = 216 for translation and rotation, 30 for scale
        all_logits_list = torch.stack(all_logits_list, dim=0) # (num_poses, num_samples * (num_regions+1), num_class) num_poses = 216 for translation and rotation, 30 for scale
        np.save(mode_folder + "region_shapley_value.npy", region_shapley_list)
        torch.save(all_logits_list, mode_folder + "all_logits.pt") # save the logits for further use
        save_info_fn(all_transform_params, mode_folder)
        t_end = time.time()
        io.cprint("time: %f" % (t_end - t_start))
        io.close()