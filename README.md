## Overview

This repository is an implementation of the paper **Interpreting Representation Quality of DNNs**
**for 3D Point Cloud Processing** ([arxiv](https://arxiv.org/abs/2111.03549)), which was accepted by NeurIPS 2021.



## Requirements

- Python 3.8
- pytorch 1.7.0/1.7.1
- CUDA 10.2
- numpy 1.19.5
- torchvision 0.8.2 
- scikit-learn 0.24.2

All models were trained on two NVIDIA TITAN RTX GPUs using `torch.nn.DataParallel`.

Besides model training, the experiments were conducted on two types of GPUs: NVIDIA TITAN RTX and NVIDIA GeForce RTX 3090. If you want to better reproduce the results, you can run on the corresponding GPUs following the table below.

|            | PointNet | PointNet++ | PointConv | DGCNN | GCNN  | adv-GCNN |
| ---------- | -------- | ---------- | --------- | ----- | ----- | -------- |
| ModelNet10 | TITAN    | TITAN      | TITAN     | 3090  | 3090  | TITAN    |
| ShapeNet   | TITAN    | 3090       | TITAN     | 3090  | TITAN | TITAN    |



## Data preparation

We use the ModelNet10 dataset and ShapeNet Part dataset. First, run the following command

```shell
mkdir data
```

### ModelNet10

The dataset is originally in CAD mesh format. We provide the dataset in numpy format, which can be downloaded from [Google Drive](https://drive.google.com/file/d/1llgbjD8XMVaw2mWtoY2RWoLQelncCgZh/view?usp=sharing). After downloading the zip file, please unzip it to the `data` directory you have just created. Please do not change the name of the folders.

### ShapeNet Part

The dataset can be downloaded [here](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip). After downloading the zip file, please unzip it to the `data` directory you have just created. Please do not change the name of the folders.

### Data directory structure

The directory should look like the following structure after you prepare the datasets.

```
|----data
	|----modelnet10_numpy
	|----shapenetcore_partanno_segmentation_benchmark_v0
```



## Usage

### Model Training

Run the following training script to train all the models:

```shell
./scripts/train_models.sh
```

You can also run a single command in the script to train a specific model. The models are by default trained on two GPUs (GPU 0 and GPU 1). The models will be saved to the folder `checkpoints`.

We also provide our pretrained models [here](https://drive.google.com/file/d/1sVSuMwMOZbO-Pn3yqYFZxDAiVMZX-zLm/view?usp=sharing).



### Sensitivity and spatial smoothness 

Run the following scripts to save the centers of the pointcloud regions generated by Farthest Point Sampling (FPS):

```shell
python final_save_fps.py --dataset=modelnet10
python final_save_fps.py --dataset=shapenet
```

Run the following scripts in order to save the Shapley value results: 

```shell
./scripts/exp_shapley.sh
```

where `model` is chosen from `[pointnet, pointnet2, pointconv, dgcnn, gcnn, gcnn_adv]`  and `dataset` is chosen from `[modelnet10, shapenet]`.  You can change the GPU id by changing the `device_id` variable in the script.

**Note**: The default batch size used in this experiment assumes that you have 24G memory on your GPU. If you want to change the batch size for calculating Shapley value, please change the numbers in `config.py`.



To view or print the results, run

```shell
python final_result.py --result_mode=<result_mode> --model=<model> --dataset=<dataset>
```

Details about the args:

- `<result_mode>` is chosen from
  - `sensitivity`: calculate the sensitivities as in Table 2
  - `correlation`: calculate the Pearson correlation coefficients as in Table 3
  - `smoothness`: calculate the non-smoothness as in Table 4
- `<model>` is chosen from `[pointnet, pointnet2, pointconv, dgcnn, gcnn, gcnn_adv, all]`. If you choose `all`, and `<result_mode>` is one of `sensitivity|correlation|smoothness` , then the script will print the result for all the models.
- `<dataset>` is chosen from `[modelnet10, shapenet]` 



### Representation complexity

To save the results of multi-order interactions, run the following script:

```shell
./scripts/exp_interaction.sh
```

where `model` is chosen from `[pointnet, pointnet2, pointconv, dgcnn, gcnn, gcnn_adv]`  and `dataset` is chosen from `[modelnet10, shapenet]`.  You can change the GPU id by changing the `device_id` variable in the script.

**Note**: The default batch size used in this experiment assumes that you have 24G memory on your GPU. If you want to change the batch size for calculating interaction, please change the numbers in `config.py`.

To generate the interaction plot as in Figure 5, run the following script:

```shell
./scripts/plot_inter.sh
```

You can view the plot in `figures_show/interaction_all/interaction_all.pdf`.



## Citation

If you use this project in your research, please cite it.

```
@misc{shen2021interpreting,
      title={Interpreting Representation Quality of DNNs for 3D Point Cloud Processing}, 
      author={Wen Shen and Qihan Ren and Dongrui Liu and Quanshi Zhang},
      year={2021},
      eprint={2111.03549},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## References

- Pytorch implementation of PointNet: [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- Pytorch implementation of PointNet++: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- DGCNN: [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)
- PointConv: [DylanWusee/pointconv](https://github.com/DylanWusee/pointconv)





