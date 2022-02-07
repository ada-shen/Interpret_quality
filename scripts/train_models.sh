#!/bin/bash
# normally trained models on ModelNet10
python main_cls_seed.py --model=pointnet --dataset=modelnet10
python main_cls_seed.py --model=pointnet2 --dataset=modelnet10
python main_cls_seed.py --model=pointconv --dataset=modelnet10
python main_cls_seed.py --model=dgcnn --dataset=modelnet10
python main_cls_seed.py --model=gcnn --dataset=modelnet10

# adv-GCNN on ModelNet10
python main_cls_seed.py --model=gcnn --dataset=modelnet10 --train_rot_all_perturbation
python main_cls_adv.py --model=gcnn --dataset=modelnet10 --train_rot_all_perturbation

# normally trained models on ShapeNet part
python main_cls_seed.py --model=pointnet --dataset=shapenet
python main_cls_seed.py --model=pointnet2 --dataset=shapenet
python main_cls_seed.py --model=pointconv --dataset=shapenet
python main_cls_seed.py --model=dgcnn --dataset=shapenet
python main_cls_seed.py --model=gcnn --dataset=shapenet

# adv-GCNN on ShapeNet part
python main_cls_seed.py --model=gcnn --dataset=shapenet --train_rot_all_perturbation
python main_cls_adv.py --model=gcnn --dataset=shapenet --train_rot_all_perturbation
