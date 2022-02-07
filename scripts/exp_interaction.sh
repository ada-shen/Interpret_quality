#!/bin/bash
model="pointnet";
dataset="shapenet";
device_id=0;
python final_gen_pair.py --model=$model --dataset=$dataset --device_id=$device_id
python final_point_binary_interaction_logits.py --model=$model --dataset=$dataset --device_id=$device_id
python final_cal_interactions.py --model=$model --dataset=$dataset --device_id=$device_id