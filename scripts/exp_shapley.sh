#!/bin/bash
model="pointnet";
dataset="shapenet";
device_id=0;
python final_shapley_value.py --model=$model --dataset=$dataset --device_id=$device_id
python final_trans_center_enum_all.py --model=$model --dataset=$dataset --device_id=$device_id
python final_rotate_center_enum_all.py --model=$model --dataset=$dataset --device_id=$device_id
python final_scale_center_enum_all.py --model=$model --dataset=$dataset --device_id=$device_id
python final_smoothness_center_enum_all.py --model=$model --dataset=$dataset --device_id=$device_id