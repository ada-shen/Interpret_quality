import os
import argparse
import time
import torch
from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test
import numpy as np
from torch.utils.data import DataLoader
from tools.final_util import IOStream, set_random, set_model_args, set_shapley_batch_size, mkdir, load_model, get_folder_name_list
from tools.final_util import NUM_POINTS, NUM_REGIONS, NUM_SAMPLES, SHAPENET_CLASS
from tools.final_common import shap_sampling_all_regions_batch


STEP = 1e-3  # step size in a single gradient descent/ascent step
ENUM_STEP = 0.05  # step size of enumeration of linearity/planarity/scattering (might contain multiple gradient decent/ascent steps)
EPOCH = 50  # max epoch to run (actually the code will finish in 1/ENUM_STEP epochs)
VAR_THRESHOLD = 0.003  # threshold of variance on the 3 principal orientations
DIST_THRESHOLD = 0.03  # threshold of distance by which each point deviates from original position
STOP_RATIO = 0.5  # if ratio of points that exceed distance bound is greater than this ratio, stop enumeration
MAX_ITERATION = 100  # max number of iterations in an enumeration step


def cal_principal_orientation(data_region_i_orig):
    """ calculate principal orientations of the original region i
    Input:
        data_region_i_orig: (S,3) tensor, original points in region i
    Return:
        o1, o2, o3: (3,) tensors, normalized (L2 norm = 1) principal orientation o1,o2,o3,
            corresponding to the largest, middle, smallest eigenvalue
    """
    S = data_region_i_orig.size()[0]
    data_mean = torch.mean(data_region_i_orig, dim=0)  # (3,)
    data_region_i_minus_mean = data_region_i_orig - data_mean  # (S,3)

    # compute eigenvectors of covairance matrix
    data_region_i_minus_mean = data_region_i_minus_mean.unsqueeze(dim=2)  # (S,3,1)
    cov_matrix = torch.bmm(data_region_i_minus_mean, data_region_i_minus_mean.transpose(1, 2))  # (S,3,1)x(S,1,3)->(S,3,3)

    # covariance matrix should be divided by S-1, not S
    cov_matrix = torch.sum(cov_matrix, dim=0) / (S-1)  # (3,3)

    eigenvalues, eigenvectors = torch.symeig(cov_matrix, eigenvectors=True)

    # lambda1, lambda2, lambda3 = eigenvalues[2], eigenvalues[1], eigenvalues[0]
    o1, o2, o3 = eigenvectors[:, 2].clone().detach(), eigenvectors[:, 1].clone().detach(), eigenvectors[:,0].clone().detach()
    return o1, o2, o3


def cal_variance(data_region_i, o1, o2, o3):
    """ calculate the variance of the projected points on each principal orientations
    Input:
        data_region_i: (S,3) tensor, points in region i
        o1, o2, o3: (3,) tensor, principal orientations
    Return:
        var1, var2, var3: scalar tensors, variance on each orientation
    """
    projection1 = torch.matmul(data_region_i, o1)
    var1 = torch.var(projection1)  # it will automatically use the unbiased estimator
    projection2 = torch.matmul(data_region_i, o2)
    var2 = torch.var(projection2)
    projection3 = torch.matmul(data_region_i, o3)
    var3 = torch.var(projection3)
    return var1, var2, var3


def apply_var_bound(var1, var2, var3, var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb):
    """ if var exceeds the bound, then don't pass gradient through it"""
    if var1 > var1_ub or var1 < var1_lb:
        var1 = var1.detach()
    if var2 > var2_ub or var2 < var2_lb:
        var2 = var2.detach()
    if var3 > var3_ub or var3 < var3_lb:
        var3 = var3.detach()
    return var1, var2, var3

def set_var_bound(var1_orig, var2_orig, var3_orig, args):
    """ returns upper bound and lower bound of the variances """
    var1_ub, var1_lb = var1_orig + args.var_threshold, var1_orig - args.var_threshold
    var2_ub, var2_lb = var2_orig + args.var_threshold, var2_orig - args.var_threshold
    var3_ub, var3_lb = var3_orig + args.var_threshold, var3_orig - args.var_threshold
    return var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb

def sort_var(var1, var2, var3):
    """ returns the sorted variances """
    var_array = np.array([var1.cpu().item(), var2.cpu().item(), var3.cpu().item()])
    sort_idx = np.argsort(var_array).tolist()
    if sort_idx == [0, 1, 2]:
        s_min, s_mid, s_max = var1, var2, var3
    elif sort_idx == [0, 2, 1]:
        s_min, s_mid, s_max = var1, var3, var2
    elif sort_idx == [1, 0, 2]:
        s_min, s_mid, s_max = var2, var1, var3
    elif sort_idx == [1, 2, 0]:
        s_min, s_mid, s_max = var2, var3, var1
    elif sort_idx == [2, 0, 1]:
        s_min, s_mid, s_max = var3, var1, var2
    else:  # sort_idx == [2,1,0]
        s_min, s_mid, s_max = var3, var2, var1
    return s_min, s_mid, s_max


def apply_distance_bound(data_region_i, data_region_i_orig, args):
    """
    Input:
        data_region_i: (S,3) tensor, current region i points
        data_region_i_orig: (S,3) tensor, original region i points
    Return:
        data_region_i: modified data_region_i
        count: number of points that exceed distance bound
    """
    with torch.no_grad():
        region_i_diff = data_region_i - data_region_i_orig #(S,3)
        region_i_diff_distance = torch.norm(region_i_diff, dim=1) #(S,)

        total_points = region_i_diff_distance.shape[0]
        count = 0
        for i in range(total_points):  # check bound
            if region_i_diff_distance[i] > args.dist_threshold:
                count += 1
                data_region_i[i].data = data_region_i_orig[i].data + args.dist_threshold * region_i_diff[i] / region_i_diff_distance[i]
    return data_region_i, count


def gradient_descent(data_region_i, objective, args):
    """
    Input:
        data_region_i: (S,3)
        objective: choice = ["inc", "dec"], indicate whether to perform gradient ascent or descent
    Return:
        (S,3), modified data_region_i
    """
    if_grad_none = False
    if data_region_i.grad != None:  # when all 3 lambdas exceed the bound, grad might be None. In this case, we do not update the region any more
        grad = data_region_i.grad.data
        norm = torch.norm(grad)
        delta = args.step * grad / norm if norm != 0 else 1e-8 # when norm=0, give a small perturbation
        data_region_i.data = data_region_i.data + delta if objective == "inc" else data_region_i.data - delta
    else:
        if_grad_none = True
    return data_region_i, if_grad_none




def cal_smoothness_orig(var1_orig, var2_orig, var3_orig, io, args):
    """
    Input:
        var1_orig, var2_orig, var3_orig: scalar tensors, original variances on the 3 principal orientations
    Return: smoothness_orig: scalar, original smoothness of this region (linearity/planarity/scattering,
        depending on args.mode)
    """
    with torch.no_grad():
        s_min, s_mid, s_max = sort_var(var1_orig, var2_orig, var3_orig)
        if args.mode == "linearity":
            smoothness_orig = (s_max - s_mid) / s_max
        elif args.mode == "planarity":
            smoothness_orig = (s_mid - s_min) / s_max
        else:  # args.mode == "scattering"
            smoothness_orig = s_min / s_max
        io.cprint("orig %s: %.8f" % (args.mode, smoothness_orig))
        return smoothness_orig.cpu().item()


def compare_smoothness(smoothness, target_smoothness, objective):
    if objective == "inc":
        return smoothness < target_smoothness
    else:
        return smoothness > target_smoothness

def check_stop_condition(count, num_total_points, if_grad_none, iteration, args, io):
    """ there are 3 conditions under which we stop the enumeration
    1. ratio of points that exceed distance bound is greater than args.stop_ratio
    2. smoothness is greater than upper bound, or lower than lower bound
    3. smoothness cannot be updated by gradient due to variance bound, i.e., if_grad_none=True
    """
    if count / num_total_points > args.stop_ratio:
        io.cprint("stop: more than 50% points exceed distance bound")
    if if_grad_none:
        io.cprint("stop: all orientations exceed variance bound, no gradient")
    if iteration > args.max_iteration:
        io.cprint("stop: achieve max iteration")
    return count/num_total_points > args.stop_ratio or if_grad_none or iteration > args.max_iteration


def update_region(data_copy, data_region_i_orig, region_id, region_i, objective, io,
                  args, orientations, bounds, smoothness_orig):
    """
    Input:
        data_copy: (B,num_points,3) tensor, B=1
        data_region_i_orig: (S,3) tensor
        region_id: (num_points,) ndarray
        region_i: scalar
        objective: choice = ["inc", "dec"], indicate whether to perform gradient ascent or descent
        orientations: 3-tuple (o1, o2, o3)
        bounds: 6-tuple (var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb)
        smoothness_orig: scalar, smoothness (linearity/planarity/scattering) of the last epoch
    Return:
        modified data_copy: (B,num_points,3) tensor, current smoothness: scalar,
        if_update: indicator of whether to update in the next epoch
    """
    o1, o2, o3 = orientations[0], orientations[1], orientations[2]
    var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb = bounds[0], bounds[1], bounds[2], bounds[3], bounds[4],bounds[5]
    smoothness = smoothness_orig
    target_smoothness = smoothness_orig + args.enum_step if objective == "inc" else smoothness_orig - args.enum_step
    io.cprint("\tregion%d orig %s: %.8f, target %s: %.8f" % (region_i, args.mode, smoothness_orig, args.mode, target_smoothness))
    if_update = True # if True, then still update this region in the next epoch, if set to False, then don't update
    iteration = 0
    while compare_smoothness(smoothness, target_smoothness, objective):
        data_region_i = data_copy[:,region_id == region_i,:].squeeze().clone().detach().requires_grad_(True)
        var1, var2, var3 = cal_variance(data_region_i, o1, o2, o3)
        var1, var2, var3 = apply_var_bound(var1, var2, var3, var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb)
        s_min, s_mid, s_max = sort_var(var1, var2, var3)
        if args.mode == "linearity":
            linearity = (s_max - s_mid) / s_max
            smoothness = linearity.cpu().item()
            if s_max.requires_grad == True or s_mid.requires_grad == True:
                linearity.backward()

        elif args.mode == "planarity":
            planarity = (s_mid - s_min) / s_max
            smoothness = planarity.cpu().item()
            if s_max.requires_grad == True or s_mid.requires_grad == True or s_min.requires_grad == True:
                planarity.backward()

        else:  # args.mode == "scattering"
            scattering = s_min / s_max
            smoothness = scattering.cpu().item()
            if s_max.requires_grad == True or s_min.requires_grad == True:
                scattering.backward()
        # io.cprint("%s: %.8f" % (args.mode, smoothness))

        data_region_i, if_grad_none = gradient_descent(data_region_i, objective, args)

        num_total_points = data_region_i.size()[0]
        data_region_i, count = apply_distance_bound(data_region_i, data_region_i_orig, args)

        data_copy[:,region_id == region_i,:] = data_region_i.unsqueeze(0).data  # (B,S,3), B=1
        iteration += 1
        if check_stop_condition(count, num_total_points, if_grad_none, iteration, args, io):
            if_update = False
            break
    io.cprint("var1: %.8f, var2: %.8f, var3: %.8f" % (var1.cpu().item(), var2.cpu().item(), var3.cpu().item()))
    io.cprint("curr smoothness: %.8f"%smoothness)
    return data_copy, smoothness, if_update


def get_original_region_info(data, region_id, region_i, io, args):
    """ return original region points, original smoothness, principal orientations and var bounds of region_i
    Input:
        data: (B,num_points,3) tensor, original point cloud
        region_id: (num_points,)
        region_i: scalar
    Return:
        data_region_i_orig: (S,3) tensor, original point coordinates of region_i
        smoothness_orig: scalar, original smoothness
        orientations: (o1, o2, o3) tuple, principal orientations
        bounds: (var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb) tuple, variance bounds
    """
    data_region_i_orig = data[:, region_id == region_i,:].squeeze().clone().detach()  # (S,3), S is number of points in this region
    o1, o2, o3 = cal_principal_orientation(data_region_i_orig)  # (3,) tensors
    var1_orig, var2_orig, var3_orig = cal_variance(data_region_i_orig, o1, o2, o3)  # scalar tensors
    io.cprint("var1 orig: %.8f, var2 orig: %.8f, var3 orig: %.8f" % (
        var1_orig.cpu().item(), var2_orig.cpu().item(), var3_orig.cpu().item()))
    var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb = set_var_bound(var1_orig, var2_orig, var3_orig, args) # scalar tensors
    smoothness_orig = cal_smoothness_orig(var1_orig, var2_orig, var3_orig, io, args)

    orientations = (o1, o2, o3)
    bounds = (var1_ub, var2_ub, var3_ub, var1_lb, var2_lb, var3_lb)
    return data_region_i_orig, smoothness_orig, orientations, bounds


def check_indicators_all_false(indicators):
    """
    Input: indicators: (num_regions,) list
    Return: True if all indicators are False, False if any indicator is True
    """
    for indicator in indicators:
        if indicator:
            return False
    return True # when all elements are False

def test_all_region(model, data, lbl, load_order_list, region_id, mode_folder, args, objective):
    """
    Input:
        data: (B,num_points,3) tensor, B=1, point cloud
        lbl: (B,) tensor, B=1, label
        load_order_list: (num_samples_save,num_regions) ndarray, saved orders
        region_id: (num_points,) ndarray, saved region id, record that each point belongs to which region
        objective: increase or decrease linearity/planarity/scattering, choice = ["inc", "dec"]
    """
    assert objective in ["inc", "dec"]
    t_start = time.time()
    data_list, smoothness_list, region_shapley_list, all_logits_list = [], [], [], []
    result_path = mode_folder + "allregion_%s/" % (objective)
    mkdir(result_path)
    io = IOStream(result_path + "log.txt")
    io.cprint(str(args))

    data_copy = data.clone().detach() #(B,num_points,3), we disturb the points on data_copy and do not modifty data itself

    # obtain original smoothness and Shapley value
    orig_shap_value, _ = shap_sampling_all_regions_batch(model, data, lbl, region_id, load_order_list, args) #(num_regions,)
    io.cprint("origin shapley of this region: %s" % str(orig_shap_value))
    np.save(result_path + "orig_shapley_value.npy", orig_shap_value) # (num_regions,)
    all_data_region_i_orig, all_smoothness_orig, all_orientations, all_bounds = [],[],[],[]
    for region_i in range(args.num_regions):
        data_region_i_orig, smoothness_orig, orientations, bounds = get_original_region_info(data, region_id, region_i,io,args)
        all_data_region_i_orig.append(data_region_i_orig)
        all_smoothness_orig.append(smoothness_orig)
        all_orientations.append(orientations)
        all_bounds.append(bounds)

    indicators = [True for _ in range(args.num_regions)] # (num_regions,) currently all True, indicate whether we can update this region
    for i in range(args.epoch): # max number of epochs
        io.cprint("\n************ epoch %d ***********" % (i))
        smoothness_list_temp = []

        # one round of update to all regions' smoothness
        for region_i in range(args.num_regions):
            smoothness = all_smoothness_orig[region_i] # initialize smoothness to be the value of the smoothness in the last epoch
            if indicators[region_i]: # when indicator is True, we update this region
                data_copy, smoothness, if_update = update_region(data_copy, all_data_region_i_orig[region_i],
                                                    region_id, region_i, objective, io, args, all_orientations[region_i],
                                                    all_bounds[region_i], all_smoothness_orig[region_i])
                all_smoothness_orig[region_i] = smoothness
                indicators[region_i] = if_update

            smoothness_list_temp.append(smoothness)  # append scalar

        smoothness_list.append(smoothness_list_temp) # append (num_regions,) list
        data_list.append(data_copy.cpu().numpy()) # append (B,num_points,3), B=1

        # calculate Shapley value
        region_shap_values, all_logits_this_pose = shap_sampling_all_regions_batch(model, data_copy, lbl, region_id, load_order_list, args) #(num_regions,)
        region_shapley_list.append(region_shap_values) # (num_regions,)
        all_logits_list.append(all_logits_this_pose)
        io.cprint("region shapley value: %s" % str(region_shap_values))
        if check_indicators_all_false(indicators):
            break

    # save results
    region_shapley_list = np.array(region_shapley_list)
    all_logits_list = torch.stack(all_logits_list, dim=0) # (num_poses, num_samples * (num_regions+1), num_class) tensor
    np.save(result_path + "region_shapley_value.npy", region_shapley_list) # (num_poses,num_regions) num_poses equals the number of epochs
    torch.save(all_logits_list, result_path + "all_logits.pt")  # save the logits for further use

    np.save(result_path + "%s.npy" % (args.mode), smoothness_list) # (num_poses,num_regions)
    np.save(result_path + "data_smoothness.npy", data_list)  # (num_poses,B,num_points,3), B=1, each state of the pointcloud
    t_end = time.time()
    io.cprint("time: %f" % (t_end - t_start))
    io.close()



def test_smoothness(args):
    """ program runner """
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
        data = data.to(args.device)
        lbl = lbl.to(args.device)

        folder_name = folder_name_list[pc_index]
        base_folder = args.exp_folder + "%s/" % folder_name
        mode_folder = base_folder + "%s_all/" % args.mode

        region_id = np.load(base_folder + "region_id.npy") #(num_points,)
        load_order_list = np.load(base_folder + "all_orders.npy") #(num_samples_save, num_regions)

        test_all_region(model, data, lbl, load_order_list, region_id, mode_folder, args, objective='inc')
        test_all_region(model, data, lbl, load_order_list, region_id, mode_folder, args, objective='dec')



def main():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
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
    args.step = STEP
    args.enum_step = ENUM_STEP
    args.epoch = EPOCH
    args.var_threshold = VAR_THRESHOLD
    args.dist_threshold = DIST_THRESHOLD
    args.stop_ratio = STOP_RATIO
    args.max_iteration = MAX_ITERATION
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)
    set_model_args(args)
    set_shapley_batch_size(args) # set different batch size for different models

    args.mode = "linearity"
    test_smoothness(args)
    args.mode = "planarity"
    test_smoothness(args)
    args.mode = "scattering"
    test_smoothness(args)


if __name__ == "__main__":
    main()