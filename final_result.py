import numpy as np
import torch
from torch.utils.data import DataLoader
import os

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatter, LogFormatterExponent, LogFormatterSciNotation, LogLocator, ScalarFormatter
from scipy.stats import pearsonr
import argparse

from final_data_shapley import ModelNet_Loader_Shapley_test, ShapeNetDataset_Shapley_test
from tools.final_util import (get_folder_name_list, mkdir, square_distance_np, ball_query,
                              BALL_QUERY_COEF)
from tools.final_util import NUM_REGIONS, NUM_POINTS, SHAPENET_CAT2ID, SHAPENET_CLASS
from tools.visulization import turbo_cmp


num_pc = 30
plt.rc('font', family='Times New Roman')

model_names = ['pointnet', 'pointnet2', 'pointconv', 'dgcnn', 'gcnn', 'gcnn_adv']
modes_all = ['rotate','trans','scale', 'linearity', 'planarity', 'scattering']


def get_exp_folder_name(model_name, dataset):
    return './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        model_name, dataset, num_points, num_regions)

def ax_3d_scatter_plot(fig, ax, data, region_color, region_id, bound, cmp, title=None, show_cbar=False, region_bold=None, size=(2,10), plot_lim=0.57):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    colors = np.zeros((num_points,))

    # assign colors to each region
    for region_i in range(num_regions):
        colors[region_id == region_i] = region_color[region_i]
    s = np.ones((data.shape[0],)) * size[0]
    if region_bold is not None:
        s[region_id == region_bold] = size[1]  # larger point size for emphasis

    sc = ax.scatter(x, y, z, c=colors, marker='.',s=s, alpha=1, cmap=cmp, norm=Normalize(vmin=bound[0], vmax=bound[1]))
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_zlim(-plot_lim, plot_lim)
    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)
    if show_cbar: # show colorbar for the subplot
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.tick_params(labelsize=8)
    return sc

def cal_mean_sv_intensity(model_name, mode):
    """ calculate mean Shapley value intensity E_j[|phi_i^(j)|] for a region
    This is used in exploring the relationship between sensitivity and mean Shapley value intensity (scatter plot)
    Only for trans, rotate, scale
    Returns:
    - An array of shape (num_pc, num_regions), element[k,i] is the mean Shapley value intensity
        of region i in sample k
    """
    assert mode == "trans" or mode == "rotate" or mode == "scale"
    all_mean_sv_intensity = []
    exp_folder = get_exp_folder_name(model_name, args.dataset)
    folder_name_list = get_folder_name_list(args)  # (30,) list
    for name in folder_name_list:
        base_folder = exp_folder + "%s/" % name
        region_shapley_values = np.load(base_folder + "%s_all/region_shapley_value.npy" % mode)  # (num_poses, num_regions)
        mean_sv_intensity = np.mean(np.abs(region_shapley_values), axis=0) # (num_regions,)
        all_mean_sv_intensity.append(mean_sv_intensity)
    return np.array(all_mean_sv_intensity) #(num_pc, num_regions)


"""Table 2 helper"""
def cal_sensitivity(base_folder, mode):
    """
    Return: sensitivity: (num_regions,), sensitivity of a single point cloud
    """
    if ('linearity' in mode) or ('planarity' in mode) or ('scattering' in mode):
        region_shapley_values_inc = np.load(base_folder + "%s_all/allregion_inc/region_shapley_value.npy" % (mode))  # (num_poses1, num_regions)
        region_shapley_values_dec = np.load(base_folder + "%s_all/allregion_dec/region_shapley_value.npy" % (mode))  # (num_poses2, num_regions)
        region_shapley_values = np.concatenate((region_shapley_values_inc, region_shapley_values_dec), axis=0)  # (num_poses1+num_poses2, num_regions)
    else:
        region_shapley_values = np.load(base_folder + "%s_all/region_shapley_value.npy" % (mode))  # (num_poses, num_regions)

    # denominator: use mean L1 norm of all sv in the enumeration process
    denominator = np.mean(np.sum(np.abs(region_shapley_values), axis=1))

    # numerator : use max_sv - min_sv (range) of each region
    max_per_region = np.max(region_shapley_values, axis=0)  # (num_regions,)
    min_per_region = np.min(region_shapley_values, axis=0)  # (num_regions,)
    range_per_region = max_per_region - min_per_region  # (num_regions,)
    sensitivity = range_per_region / denominator # (num_regions,)
    return sensitivity


""" Table 2 """
def cal_sensitivity_all_pc(model_name, mode):
    """ calculate sensitivity for all samples
    Return:
        all_sensitivity: An array of shape (num_sample, num_regions), element[k,i] is the normalized range
        of region i in sample k
    """
    exp_folder = get_exp_folder_name(model_name, args.dataset)
    folder_name_list = get_folder_name_list(args)  # (30,) list
    all_sensitivity = []
    for i, name in enumerate(folder_name_list):
        base_folder = exp_folder + "%s/" % name
        sensitivity = cal_sensitivity(base_folder, mode)
        # print(sensitivity.mean())
        all_sensitivity.append(sensitivity)
    return np.array(all_sensitivity) # (num_pc, num_regions)


"""Table 3"""
def cal_correlation_coef(model_name, mode):
    """ For a given model and a given mode, calculate the Pearson r coefficient of the scatter plots for all samples
        Print the mean and std of the coefficients
    """
    assert mode == "trans" or mode == "rotate" or mode == "scale"
    print("mode: %s, model: %s"%(mode, model_name))
    all_sensitivity = cal_sensitivity_all_pc(model_name=model_name, mode=mode)  # (num_pc, num_regions)
    all_mean_sv_intensity = cal_mean_sv_intensity(model_name=model_name, mode=mode)  # (num_pc, num_regions)

    all_pearson_r = []
    for i in range(num_pc):
        sensitivity, mean_sv_intensity = all_sensitivity[i,:], all_mean_sv_intensity[i,:]
        r,_ = pearsonr(sensitivity, mean_sv_intensity)
        all_pearson_r.append(r)
    all_pearson_r = np.array(all_pearson_r)
    print("mean Pearson r=%f±%f"%(all_pearson_r.mean(), all_pearson_r.std(ddof=1)))
    return all_pearson_r.mean()


"""Table 4 helper"""
def cal_shapley_smoothness_metric_single_pc(data, region_shapley_values, region_id):
    """
    Input:
        data: (N,d), N=1024, d=3
        fps_index: (num_regions,), num_regions=32
        region_shapley_values: (num_poses, num_regions), num_pose=216, num_regions=32
        region_id: (num_points), num_points=1024
    Return:
        metric: spatial smoothness metric of this sample
        metric_all_poses: spatial smoothness metric of this sample, not yet averaged among all poses
        denominator: normalization factor for this sample
    """
    num_poses = region_shapley_values.shape[0]
    region_centers = np.zeros((num_regions, 3))
    for i in range(num_regions):
        region_centers[i] = data[region_id == i].mean(axis=0)  # (3,)

    pairwise_distance = square_distance_np(data) # (num_points,num_points)
    diameter = np.sqrt(np.maximum(pairwise_distance,0)).max()
    neighbor_idx = ball_query(region_centers, r=BALL_QUERY_COEF * diameter) # use ball query to determine the neighbors, here neighbor includes region i itself

    denominator = np.abs(np.sum(region_shapley_values, axis=1)).mean()  # (num_poses,num_regions)->(num_poses,)->scalar, mean of |f(N)-f(empty)|

    all_fraction = np.zeros((num_poses, num_regions)) # (num_poses,num_regions)
    for p in range(num_poses):
        for i in range(num_regions):
            numerator = np.abs(region_shapley_values[p, i] - region_shapley_values[p, neighbor_idx[i]]).mean() # for a single region i
            fraction = numerator / denominator
            all_fraction[p, i] = fraction

    metric = all_fraction.mean() # scalar
    metric_all_poses = all_fraction.mean(axis=1) # (num_poses,)
    return metric, metric_all_poses, denominator

"""Table 4"""
def cal_shapley_smoothness_metric(model_name, mode):
    """ For a given model and a given mode, calculate the spatial smoothness for all samples
            Print the mean and std of the spatial smoothness metric
    """
    assert mode == "trans" or mode == "rotate"
    metric_all_samples = []
    exp_folder = get_exp_folder_name(model_name, args.dataset)
    if args.dataset == "modelnet10":
        data_loader = DataLoader(ModelNet_Loader_Shapley_test(args, partition='train', num_points=num_points),
                                 num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "shapenet":
        data_loader = DataLoader(ShapeNetDataset_Shapley_test(args, split='train', npoints=num_points,
                                                              class_choice=SHAPENET_CLASS, classification=True),
                                 num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("Dataset does not exist")
    folder_name_list = get_folder_name_list(args)  # (30,) list
    for pc_idx, (data, lbl) in enumerate(data_loader):
        name = folder_name_list[pc_idx]
        if name[:5] == "Knife": # skip the knife category for Shapenet, see the paper for the reason
            continue
        base_folder = exp_folder + "%s/" % name
        data = data.cpu().numpy().squeeze()  # (num_points, 3) num_points=1024
        region_id = np.load(base_folder + "region_id.npy")  # (num_points,)
        region_shapley_values = np.load(base_folder + "%s_all/region_shapley_value.npy" % mode)  # (num_poses,num_regions)

        metric, metric_all_poses,_ = cal_shapley_smoothness_metric_single_pc(data, region_shapley_values, region_id)
        print("%s, metric=%f"%(name, metric))
        metric_all_samples.append(metric)
    metric_all_samples = np.array(metric_all_samples)
    print("%s, %s, metric=%f±%f"%(model_name, mode, metric_all_samples.mean(), metric_all_samples.std(ddof=1)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='all', choices=[
        "pointnet","pointnet2","dgcnn","gcnn","pointconv","gcnn_adv","all"
    ])
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['modelnet10','shapenet','modelnet40'])
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_regions', type=int, default=32)
    parser.add_argument('--result_mode', type=str, default="sensitivity",
                        choices=["sensitivity","correlation","smoothness"])
    args = parser.parse_args()

    num_points = args.num_points
    num_regions = args.num_regions

    model_list = model_names if args.model == "all" else [args.model]

    # Table 1: calculate sensitivity
    if args.result_mode == "sensitivity":
        for model_name in model_list:
            for mode in modes_all:
                print("***\n***\n***\nmodel: %s,  exp: %s"%(model_name, mode))
                all_sensitivity = cal_sensitivity_all_pc(model_name=model_name, mode=mode)
                print("mean normalized range over all samples: %.6f±%.6f" % (all_sensitivity.mean(), all_sensitivity.std(ddof=1)))

    # Table 2: correlation between sensitivity and attribution
    elif args.result_mode == "correlation":
        for model_name in model_list:
            for mode in ["trans", "rotate","scale"]:
                cal_correlation_coef(model_name=model_name, mode=mode)

    # Table 3: calculate smoothness of Shapley values for adjacent regions
    elif args.result_mode == "smoothness":
        for model_name in model_list:
            for mode in ['trans','rotate']:
                cal_shapley_smoothness_metric(model_name, mode)

    else:
        raise Exception(f"result mode [{args.result_mode}] not implemented")

