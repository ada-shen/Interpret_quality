import sys
import numpy as np
import argparse
import os

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
from matplotlib import rcParams
from tools.final_util import get_folder_name_list, mkdir
from tools.final_util import NUM_POINTS, NUM_REGIONS, SHAPENET_INTER_SELECTED_SAMPLE, MODELNET_INTER_SELECTED_SAMPLE

plt.rc('font',family='Times New Roman')
config = {
    "mathtext.fontset":'stix',
}
rcParams.update(config)
font_size = 33
model_names = ["pointnet", "pointnet2", "pointconv", "dgcnn", "gcnn", "gcnn_adv"]

def get_interaction_normal_adv_pose(args):
    print("\n#### get interaction ####")
    all_mean_inter_normal, all_abs_mean_inter_normal, all_mean_inter_adv, all_abs_mean_inter_adv = [],[],[],[]
    for i in selected_sample_idx:
        name = folder_name_list[i]
        print("======= %s ========" % name)
        base_folder = args.exp_folder + "%s/" % name
        interaction_folder = base_folder + "interaction_seed%d/" % args.gen_pair_seed

        orders_mean_inter_normal, orders_abs_mean_inter_normal, orders_mean_inter_adv, orders_abs_mean_inter_adv = [],[],[],[]
        for ratio in args.ratios:
            orders_interaction_normal = np.load(interaction_folder + "normal/ratio%d_%s_interaction.npy" % (
                int(ratio * 100), args.output_type))  # (num_pairs, num_context)
            orders_interaction_adv = np.load(interaction_folder + "%s_adv/ratio%d_%s_interaction.npy" % (args.mode,
                int(ratio * 100), args.output_type))  # (num_pairs, num_context)

            mean_inter_normal = orders_interaction_normal.mean()  # scalar
            abs_mean_inter_normal = np.abs(orders_interaction_normal.mean(axis=1)).mean()  # scalar
            mean_inter_adv = orders_interaction_adv.mean()  # scalar
            abs_mean_inter_adv = np.abs(orders_interaction_adv.mean(axis=1)).mean()  # scalar

            orders_mean_inter_normal.append(mean_inter_normal)
            orders_abs_mean_inter_normal.append(abs_mean_inter_normal)
            orders_mean_inter_adv.append(mean_inter_adv)
            orders_abs_mean_inter_adv.append(abs_mean_inter_adv)

        all_mean_inter_normal.append(orders_mean_inter_normal)
        all_abs_mean_inter_normal.append(orders_abs_mean_inter_normal)
        all_mean_inter_adv.append(orders_mean_inter_adv)
        all_abs_mean_inter_adv.append(orders_abs_mean_inter_adv)

    return np.array(all_mean_inter_normal), np.array(all_abs_mean_inter_normal), \
           np.array(all_mean_inter_adv), np.array(all_abs_mean_inter_adv) # (num_pc, num_ratios)

def get_interaction_max_min_pose(args):
    print("\n#### get interaction ####")
    all_mean_inter, all_abs_mean_inter = [], []
    for i in selected_sample_idx:
        name = folder_name_list[i]
        print("======= %s ========" % name)
        base_folder = args.exp_folder + "%s/" % name
        interaction_folder = base_folder + "interaction_seed%d/" % args.gen_pair_seed
        single_region_folder = interaction_folder + "%s_adv_single_region/" % args.mode

        pose_mean_inter, pose_abs_mean_inter = [], []
        for region_folder_name in sorted(os.listdir(single_region_folder)):
            if not os.path.isdir(single_region_folder + region_folder_name):
                continue
            print("----- %s ------" % (region_folder_name))
            range_rank = int(region_folder_name[10:12])  # get range rank information from folder name, 1-based rank
            if range_rank != 1:
                continue
            region_folder = single_region_folder + region_folder_name + "/"

            orders_mean_inter_normal, orders_abs_mean_inter_normal = [],[]
            for ratio in args.ratios:
                orders_interaction_normal = np.load(region_folder + "normal/ratio%d_%s_interaction.npy" % (
                    int(ratio * 100), args.output_type))  # (num_pairs, num_context) interaction of a single region and its neighbor

                mean_inter_normal = orders_interaction_normal.mean()  # scalar
                abs_mean_inter_normal = np.abs(orders_interaction_normal.mean(axis=1)).mean()  # scalar

                orders_mean_inter_normal.append(mean_inter_normal)
                orders_abs_mean_inter_normal.append(abs_mean_inter_normal)

            pose_mean_inter.append(orders_mean_inter_normal)
            pose_abs_mean_inter.append(orders_abs_mean_inter_normal)

        all_mean_inter.append(pose_mean_inter)
        all_abs_mean_inter.append(pose_abs_mean_inter)

    return np.array(all_mean_inter), np.array(all_abs_mean_inter) # (num_pc, 1, num_ratios), interaction of the most sensitive region at normal pose


def ax_bar_plot(ax, orders, interaction, title=None):
    bar_width = 0.04
    ax.bar(orders, interaction, bar_width)
    ax.set_xlabel("order",fontsize=font_size,labelpad = 0)
    ax.set_ylabel("interaction",fontsize=font_size,labelpad = 0)
    x = np.array([0,1.2])
    ax.set_xticks(x)
    ax.set_xticklabels(['0', 'n-2'])
    ax.tick_params(labelsize=font_size)

    if title is not None:
        ax.set_title(title)

def ax_bar_plot_double(ax, orders, interaction_normal, interaction_adv, title=None, labels=None, color2=None):
    bar_width = 0.035
    if title is not None:
        ax.set_title(title)
    if labels is not None:
        ax.bar(orders, interaction_normal, bar_width, label=labels[0]) # label="$I^{(m)}_{nor}$")
        if color2 is not None:
            ax.bar(orders+bar_width+0.005, interaction_adv, bar_width, label=labels[1], color=color2) # label="$I^{(m)}_{adv}$")
        else:
            ax.bar(orders + bar_width + 0.005, interaction_adv, bar_width, label=labels[1],)  # label="$I^{(m)}_{adv}$")
        ax.legend()
    else:
        ax.bar(orders, interaction_normal, bar_width)
        if color2 is not None:
            ax.bar(orders + bar_width + 0.005, interaction_adv, bar_width, color=color2)
        else:
            ax.bar(orders + bar_width + 0.005, interaction_adv, bar_width)

    ax.set_xlabel("order",fontsize=font_size,labelpad = -25)
    ax.set_ylabel("interaction",fontsize=font_size,labelpad = 0)
    x = np.array([0,1.2])
    ax.set_xticks(x+bar_width/2+0.0025)
    ax.set_xticklabels(['0', 'n-2'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=font_size)


def plot_inter_single_region_vs_normal_avg(args):
    mean_inter_single_region, abs_mean_inter_single_region = get_interaction_max_min_pose(args) #  (num_pc, 1, num_ratios)
    mean_inter_normal, abs_mean_inter_normal, mean_inter_adv, abs_mean_inter_adv = get_interaction_normal_adv_pose(args) # (num_pc, num_ratios)
    save_dir = "figures/interaction_final_%s/" % args.dataset
    mkdir(save_dir)
    np.save(save_dir + "%s_%s_mean_inter_single_region.npy" % (args.model, args.dataset), mean_inter_single_region)
    np.save(save_dir + "%s_%s_abs_mean_inter_single_region.npy" % (args.model, args.dataset), abs_mean_inter_single_region)
    np.save(save_dir + "%s_%s_mean_inter_normal.npy" % (args.model, args.dataset), mean_inter_normal)
    np.save(save_dir + "%s_%s_abs_mean_inter_normal.npy" % (args.model, args.dataset), abs_mean_inter_normal)
    np.save(save_dir + "%s_%s_mean_inter_adv.npy" % (args.model, args.dataset), mean_inter_adv)
    np.save(save_dir + "%s_%s_abs_mean_inter_adv.npy" % (args.model, args.dataset), abs_mean_inter_adv)

    print("shape: ", mean_inter_single_region.shape)
    orders = np.arange(0,1.3,0.1)

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax_bar_plot_double(ax, orders,np.abs(mean_inter_normal).mean(axis=0), np.abs(mean_inter_single_region[:, 0, :]).mean(axis=0), color2='y')
    fig.subplots_adjust(top=0.55, bottom=0.2, right=0.95, left=0.35)

    plt.savefig(
        save_dir + "single_region_top_range_compare_%s_%s_%s_seed%d_all_pc.png" % (
            args.model, args.mode, args.output_type, args.gen_pair_seed))
    plt.close()


def plot_inter_normal_adv_pose(args):
    mean_inter_normal, abs_mean_inter_normal, mean_inter_adv, abs_mean_inter_adv = get_interaction_normal_adv_pose(args) # (num_pc, num_ratios)
    print(mean_inter_normal.shape)
    orders = np.arange(0,1.3,0.1)

    fig = plt.figure(figsize=(5, 5),dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    # $\mathbb{E}_{X\in \mathcal{X}} |\mathbb{E}_{i,j} [I_{ij}^{(m)}]| $
    ax_bar_plot_double(ax, orders, np.abs(mean_inter_normal).mean(axis=0), np.abs(mean_inter_adv).mean(axis=0))
    plt.subplots_adjust(top=0.55, bottom=0.2, right=0.95, left=0.35)

    save_dir = "figures/interaction_final_%s/" % args.dataset
    mkdir(save_dir)
    plt.savefig(
        save_dir + "global_in_one_%s_%s_%s_seed%d_all_pc.png" % (args.model, args.mode, args.output_type, args.gen_pair_seed))
    plt.close()


def ax_bar_plot_double_for_all(ax, orders, interaction1, interaction2, title=None, color2=None,
                               show_legend=False, label=None):
    bar_width = 0.03
    if title is not None:
        ax.set_title(title, fontsize=font_size, y=1.1)

    if label is not None:
        ax.bar(orders, interaction1, bar_width, color="#4169E1", label=label[0])
        if color2 is not None:
            ax.bar(orders + bar_width + 0.006, interaction2, bar_width, color=color2,label=label[1])
        else:
            ax.bar(orders + bar_width + 0.006, interaction2, bar_width, label=label[1])
    else:
        ax.bar(orders, interaction1, bar_width, color="#4169E1")
        if color2 is not None:
            ax.bar(orders + bar_width + 0.006, interaction2, bar_width, color=color2)
        else:
            ax.bar(orders + bar_width + 0.006, interaction2, bar_width)

    # ax.set_xlabel("order m",fontsize=font_size,labelpad = -20)
    ax.set_ylabel("$I^{(m)}$", fontsize=font_size-5, labelpad=-5)
    x = np.array([0,1.2])
    ax.set_xticks(x+bar_width/2+0.003)
    ax.set_xticklabels(['0', 'n-2'])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=font_size)
    if show_legend and label is not None:
        ax.legend(loc=7, bbox_to_anchor=(1.02, 1.2), borderaxespad=0., fancybox=False, frameon=False, mode="expand",
                  labelspacing=1, fontsize=font_size, handlelength=1, handletextpad=0.3)


def plot_inter_in_one():
    orders = np.arange(0,1.3,0.1)
    model_names_show = ["PointNet", "PointNet++", "PointConv", "DGCNN", "GCNN", "adv-GCNN"]
    data = {"modelnet10":{"normal":[], "adv":[], "single_region":[]},
            "shapenet":{"normal":[], "adv":[], "single_region":[]}
            }
    for dataset in ["modelnet10", "shapenet"]:
        for model_name in model_names:
            save_dir = "figures/interaction_final_%s/" % dataset
            mean_inter_normal = np.load(save_dir + "%s_%s_mean_inter_normal.npy" % (model_name, dataset))
            mean_inter_adv = np.load(save_dir + "%s_%s_mean_inter_adv.npy" % (model_name, dataset))
            mean_inter_single_region = np.load(save_dir + "%s_%s_mean_inter_single_region.npy" % (model_name, dataset))

            data[dataset]["normal"].append(np.abs(mean_inter_normal).mean(axis=0))
            data[dataset]["adv"].append(np.abs(mean_inter_adv).mean(axis=0))
            data[dataset]["single_region"].append(np.abs(mean_inter_single_region[:, 0, :]).mean(axis=0))

    fig = plt.figure(figsize=(30, 9), dpi=100)
    ax_dataset = fig.add_axes([0.002, 0, 0.102, 1])
    ax_dataset.spines['top'].set_visible(False)
    ax_dataset.spines['right'].set_visible(False)
    ax_dataset.spines['bottom'].set_visible(False)
    ax_dataset.spines['left'].set_visible(False)
    ax_dataset.set_axis_off()
    rect1 = patches.Rectangle(xy=(0.65, 0.73), width=0.4, height=0.23, color="#D8BFD8")
    rect2 = patches.Rectangle(xy=(0.65, 0.51), width=0.4, height=0.2, color="#D8BFD8")
    rect3 = patches.Rectangle(xy=(0.65, 0.23), width=0.4, height=0.23, color="#D8BFD8")
    rect4 = patches.Rectangle(xy=(0.65, 0.01), width=0.4, height=0.2, color="#D8BFD8")
    ax_dataset.add_patch(rect1)
    ax_dataset.add_patch(rect2)
    ax_dataset.add_patch(rect3)
    ax_dataset.add_patch(rect4)
    ax_dataset.text(x=0.76, y=0.735, s="ModelNet10", ha="left", va="bottom", fontsize=font_size-5, rotation=90)
    ax_dataset.text(x=0.76, y=0.53, s="ShapeNet", ha="left", va="bottom", fontsize=font_size-5, rotation=90)
    ax_dataset.text(x=0.76, y=0.235, s="ModelNet10", ha="left", va="bottom", fontsize=font_size-5, rotation=90)
    ax_dataset.text(x=0.76, y=0.03, s="ShapeNet", ha="left", va="bottom", fontsize=font_size-5, rotation=90)

    ax_legend1 = fig.add_axes([0.2, 0.95, 0.6, 0.05])
    ax_legend1.spines['top'].set_visible(False)
    ax_legend1.spines['right'].set_visible(False)
    ax_legend1.spines['bottom'].set_visible(False)
    ax_legend1.spines['left'].set_visible(False)
    ax_legend1.set_axis_off()
    legend1 = patches.Rectangle(xy=(0,0), width=0.06, height=0.7, color="#4169E1")
    legend2 = patches.Rectangle(xy=(0.3,0), width=0.06, height=0.7, color="#FF7F24")
    ax_legend1.add_patch(legend1)
    ax_legend1.add_patch(legend2)
    ax_legend1.text(x=0.08, y=0, s="normal samples", ha="left", va="bottom", fontsize=font_size)
    ax_legend1.text(x=0.38, y=0, s="adversarial samples (using rotations for attack, instead of perturbations)", ha="left", va="bottom", fontsize=font_size)


    ax_legend2 = fig.add_axes([0.2, 0.45, 0.6, 0.05])
    ax_legend2.spines['top'].set_visible(False)
    ax_legend2.spines['right'].set_visible(False)
    ax_legend2.spines['bottom'].set_visible(False)
    ax_legend2.spines['left'].set_visible(False)
    ax_legend2.set_axis_off()
    legend1 = patches.Rectangle(xy=(0,0), width=0.06, height=0.7, color="#4169E1")
    legend2 = patches.Rectangle(xy=(0.3,0), width=0.06, height=0.7, color="#A2CD5A")
    ax_legend2.add_patch(legend1)
    ax_legend2.add_patch(legend2)
    ax_legend2.text(x=0.08, y=0, s="among all regions", ha="left", va="bottom", fontsize=font_size)
    ax_legend2.text(x=0.38, y=0, s="among most rotation-sensitive regions", ha="left", va="bottom", fontsize=font_size)



    for i, model_name in enumerate(model_names_show):
        ax = fig.add_axes([0.16 + 0.145*i, 0.75, 0.085, 0.125])
        ax_bar_plot_double_for_all(ax, orders, data["modelnet10"]["normal"][i], data["modelnet10"]["adv"][i], title=model_name,color2="#FF7F24")
    for i, model_name in enumerate(model_names_show):
        ax = fig.add_axes([0.16 + 0.145*i, 0.55, 0.085, 0.125])
        ax_bar_plot_double_for_all(ax, orders, data["shapenet"]["normal"][i], data["shapenet"]["adv"][i], color2='#FF7F24')

    for i, model_name in enumerate(model_names_show):
        ax = fig.add_axes([0.16 + 0.145*i, 0.26, 0.085, 0.125])
        ax_bar_plot_double_for_all(ax, orders, data["modelnet10"]["normal"][i], data["modelnet10"]["single_region"][i], title=model_name,color2="#A2CD5A")

    for i, model_name in enumerate(model_names_show):
        ax = fig.add_axes([0.16 + 0.145*i, 0.06, 0.085, 0.125])
        ax_bar_plot_double_for_all(ax, orders, data["shapenet"]["normal"][i], data["shapenet"]["single_region"][i], color2='#A2CD5A')

    for i in range(6):
        fig.text(x=0.185 + 0.145*i,y=0.695,s="order",ha="left",va="bottom",fontsize=font_size)
        fig.text(x=0.185 + 0.145*i,y=0.495,s="order",ha="left",va="bottom",fontsize=font_size)
        fig.text(x=0.185 + 0.145*i,y=0.205,s="order",ha="left",va="bottom",fontsize=font_size)
        fig.text(x=0.185 + 0.145*i,y=0.005,s="order",ha="left",va="bottom",fontsize=font_size)

    fig.text(x=0.04,y=0.7,s="(a)",ha="left",va="bottom",fontsize=font_size+5)
    fig.text(x=0.04, y=0.2, s="(b)", ha="left", va="bottom", fontsize=font_size+5)

    save_dir = "figures_show/interaction_all/"
    mkdir(save_dir)
    plt.savefig(save_dir + "interaction_all.pdf")
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcnn_adv',
                        choices=['pointnet', 'pointnet2', 'pointconv', 'dgcnn', 'gcnn', 'gcnn_adv'])
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N',choices=['modelnet10', 'shapenet'])
    parser.add_argument("--ratios", default=[0., 0.04, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], type=list)
    parser.add_argument('--gen_pair_seed', type=int, default=1, help='seed used in gen_pair, only used for checking instability')

    parser.add_argument('--mode', type=str, default='rotate')

    parser.add_argument('--output_type', default='pred', type=str, choices=["gt", "pred"])
    parser.add_argument("--num_pairs_random", default=300, type=int)  # number of random pairs when gen_pair_type is random
    parser.add_argument("--num_save_context_max", default=100, type=int)  # # max number of contexts for each I_ij
    parser.add_argument("--plot_mode", default="all", type=str,
                        choices=["all","single_region_vs_normal_avg","normal_vs_adv"])
    args = parser.parse_args()


    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    args.exp_folder = './checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_REGIONNUM_%d_shapley_test/' % (
        args.model, args.dataset, args.num_points, args.num_regions)

    folder_name_list = get_folder_name_list(args)
    if args.dataset == "modelnet10":
        selected_sample_idx = MODELNET_INTER_SELECTED_SAMPLE
    else:
        selected_sample_idx = SHAPENET_INTER_SELECTED_SAMPLE

    if args.plot_mode == "normal_vs_adv":
        plot_inter_normal_adv_pose(args)
    elif args.plot_mode == "single_region_vs_normal_avg":
        plot_inter_single_region_vs_normal_avg(args)
    elif args.plot_mode == "all":
        plot_inter_in_one()
    else:
        raise Exception(f"plot_mode [{args.mode}] not implemented")





