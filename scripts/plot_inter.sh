#!/bin/bash
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=shapenet --model=pointnet
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=shapenet --model=pointnet2
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=shapenet --model=pointconv
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=shapenet --model=dgcnn
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=shapenet --model=gcnn
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=shapenet --model=gcnn_adv

python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=modelnet10 --model=pointnet
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=modelnet10 --model=pointnet2
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=modelnet10 --model=pointconv
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=modelnet10 --model=dgcnn
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=modelnet10 --model=gcnn
python plot_interaction.py --plot_mode=single_region_vs_normal_avg --mode=rotate --dataset=modelnet10 --model=gcnn_adv

python plot_interaction.py --plot_mode=all --mode=rotate
