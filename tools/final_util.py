import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.pointnet import PointNetCls
from models.dgcnn import DGCNN_cls, GCNN_cls
from models.pointnet2 import PointNet2ClsMsg
from models.pointconv import PointConvDensityClsSsg
from collections import OrderedDict
import json
from torch.autograd import Function
from config import CONFIG

NUM_POINTS = 1024  # number of points in each point cloud
NUM_REGIONS = 32  # number of regions in each point cloud
NUM_SAMPLES_SAVE = 1000  # number of random permutations to save at initial state
NUM_SAMPLES = 100  # number of random permutations to calculate Shapley value
K_FOR_DGCNN = 20

# test samples for Shapley value
DATA_MODELNET_SHAPLEY_TEST = 'modelnet10_train_final30.txt'
DATA_SHAPENET_SHAPLEY_TEST = 'shapenet_train_selected.json'

# test samples for interaction
MODELNET_INTER_SELECTED_SAMPLE = [0,3,6,9,12,15,18,21,24,27]
SHAPENET_INTER_SELECTED_SAMPLE = [0,3,6,9,12,15,19,21,24,27]


SHAPENET_CLASS = ["Bag", "Cap", "Earphone", "Knife", "Laptop", "Motorbike", "Mug", "Pistol", "Rocket", "Skateboard"] # 10 classes out of 16
SHAPENET_ID2CAT = {
"02691156":"Airplane",
"02773838":"Bag", #
"02954340":"Cap", #
"02958343":"Car",
"03001627":"Chair",
"03261776":"Earphone", #
"03467517":"Guitar",
"03624134":"Knife", #
"03636649":"Lamp",
"03642806":"Laptop", #
"03790512":"Motorbike", #
"03797390":"Mug", #
"03948459":"Pistol", #
"04099429":"Rocket", #
"04225987":"Skateboard", #
"04379243":"Table"
}
SHAPENET_CAT2ID = {v: k for k, v in SHAPENET_ID2CAT.items()}

# modelnet10 model
MODEL_PATH_MODELNET_POINTNET = "checkpoints/exp_MODEL_pointnet_DATA_modelnet10_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_MODELNET_POINTNET2 = "checkpoints/exp_MODEL_pointnet2_DATA_modelnet10_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_MODELNET_POINTCONV = "checkpoints/exp_MODEL_pointconv_DATA_modelnet10_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_MODELNET_DGCNN = "checkpoints/exp_MODEL_dgcnn_DATA_modelnet10_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_MODELNET_GCNN = "checkpoints/exp_MODEL_gcnn_DATA_modelnet10_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_MODELNET_GCNN_ADV = "checkpoints/exp_MODEL_gcnn_adv_DATA_modelnet10_POINTNUM_1024_clean_with_all_rot_da/models/model_399.t7"


# shapenet model
MODEL_PATH_SHAPENET_POINTNET = "checkpoints/exp_MODEL_pointnet_DATA_shapenet_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_SHAPENET_POINTNET2 = "checkpoints/exp_MODEL_pointnet2_DATA_shapenet_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_SHAPENET_POINTCONV = "checkpoints/exp_MODEL_pointconv_DATA_shapenet_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_SHAPENET_DGCNN = "checkpoints/exp_MODEL_dgcnn_DATA_shapenet_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_SHAPENET_GCNN = "checkpoints/exp_MODEL_gcnn_DATA_shapenet_POINTNUM_1024_clean/models/model_best.t7"
MODEL_PATH_SHAPENET_GCNN_ADV = "checkpoints/exp_MODEL_gcnn_adv_DATA_shapenet_POINTNUM_1024_clean_with_all_rot_da/models/model_399.t7"

BALL_QUERY_COEF = 0.25


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def cal_rank(values):
    sort_idx = np.argsort(values)
    rank = np.argsort(sort_idx)
    return rank


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def square_distance_np(x):
    """
    Inputs:
        x: A numpy array of shape (N, F)
    Returns:
        A numpy array D of shape (N, N) where D[i, j] is the squared Euclidean distance
        between x[i] and x[i].
    """
    xx = np.sum(x ** 2, axis=1, keepdims=True)  # (N,1)
    D = xx + xx.T - 2 * np.matmul(x, x.T)
    return D

def square_distance(src, dst):
    """ Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def ball_query(x, r):
    """
    Input:
        x: (num_regions,d) ndarray, num_regions=32, d=3
        r: scalar, radius of ball query
    Return:
        mask: (num_regions, num_regions), entry is True if pairwise_distance_sq[i,j] < r^2, False otherwise
    """
    pairwise_distance_sq = square_distance_np(x)
    mask = (pairwise_distance_sq < r**2)
    return mask

def set_model_args(args):
    if args.dataset == "modelnet10":
        if args.model == "dgcnn":
            args.k = K_FOR_DGCNN
            args.model_path = MODEL_PATH_MODELNET_DGCNN
        elif args.model == "gcnn":
            args.k = K_FOR_DGCNN
            args.model_path = MODEL_PATH_MODELNET_GCNN
        elif args.model == "pointnet":
            args.feature_transform = True
            args.model_path = MODEL_PATH_MODELNET_POINTNET
        elif args.model == "pointnet2":
            args.model_path = MODEL_PATH_MODELNET_POINTNET2
        elif args.model == "pointconv":
            args.model_path = MODEL_PATH_MODELNET_POINTCONV
        elif args.model == "gcnn_adv":
            args.k = K_FOR_DGCNN
            args.model_path = MODEL_PATH_MODELNET_GCNN_ADV
        else:
            raise Exception("Model not implemented")

    elif args.dataset == "shapenet":
        if args.model == "dgcnn":
            args.k = K_FOR_DGCNN
            args.model_path = MODEL_PATH_SHAPENET_DGCNN
        elif args.model == "gcnn":
            args.k = K_FOR_DGCNN
            args.model_path = MODEL_PATH_SHAPENET_GCNN
        elif args.model == "pointnet":
            args.feature_transform = True
            args.model_path = MODEL_PATH_SHAPENET_POINTNET
        elif args.model == "pointnet2":
            args.model_path = MODEL_PATH_SHAPENET_POINTNET2
        elif args.model == "pointconv":
            args.model_path = MODEL_PATH_SHAPENET_POINTCONV
        elif args.model == "gcnn_adv":
            args.k = K_FOR_DGCNN
            args.model_path = MODEL_PATH_SHAPENET_GCNN_ADV
        else:
            raise Exception("Model not implemented")

    else:
        raise Exception("Dataset does not exist")


def set_shapley_batch_size(args):
    if args.model == 'pointnet2':
        args.shapley_batch_size = CONFIG["shapley_batch_size"]["pointnet2"]
    elif args.model == 'pointnet':
        args.shapley_batch_size = CONFIG["shapley_batch_size"]["pointnet"]
    elif args.model == 'dgcnn':
        args.shapley_batch_size = CONFIG["shapley_batch_size"]["dgcnn"]
    elif args.model == "gcnn" or args.model == 'gcnn_adv':
        args.shapley_batch_size = CONFIG["shapley_batch_size"]["gcnn"]
    elif  args.model == "pointconv":
        args.shapley_batch_size = CONFIG["shapley_batch_size"]["pointconv"]
    else:
        raise Exception("Not implemented")

def set_interaction_batch_size(args):
    if args.model == 'pointnet2':
        args.interaction_batch_size = CONFIG["interaction_batch_size"]["pointnet2"]
    elif args.model == 'pointnet':
        args.interaction_batch_size = CONFIG["interaction_batch_size"]["pointnet"]
    elif args.model == 'dgcnn':
        args.interaction_batch_size = CONFIG["interaction_batch_size"]["dgcnn"]
    elif args.model == "gcnn" or args.model == 'gcnn_adv':
        args.interaction_batch_size = CONFIG["interaction_batch_size"]["gcnn"]
    elif  args.model == "pointconv":
        args.interaction_batch_size = CONFIG["interaction_batch_size"]["pointconv"]
    else:
        raise Exception("Not implemented")


def load_model(args):
    if args.model == 'pointnet2':
        model = PointNet2ClsMsg(args).to(args.device)
    elif args.model == 'pointnet':
        model = PointNetCls(args).to(args.device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(args.device)
    elif args.model == "gcnn" or args.model == 'gcnn_adv':
        model = GCNN_cls(args).to(args.device)
    elif args.model == "pointconv":
        model = PointConvDensityClsSsg(args).to(args.device)
    else:
        raise Exception("Not implemented")

    state_dict = torch.load(args.model_path, map_location=args.device)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module.' in k:  # for model trained on multiple gpus
            name = k[len('module.'):]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.eval()
    return model


def get_folder_name_list(args):
    """ get names of all samples for Shapley value test
    """
    if args.dataset == "modelnet10":
        f = open(os.path.join('misc', DATA_MODELNET_SHAPLEY_TEST), 'r')
        names = [str.rstrip() for str in f.readlines()]
        f.close()
    elif args.dataset == "shapenet":
        splitfile = os.path.join('misc', DATA_SHAPENET_SHAPLEY_TEST)
        filelist = json.load(open(splitfile, 'r'))
        names = []
        for file in filelist:
            _, category, uuid = file.split('/')
            names.append(SHAPENET_ID2CAT[category] + "_" + uuid) # format: classname_uuid

    else:
        raise Exception("Dataset does not exist")

    return names



class rot_angle_axis(Function):

    @staticmethod
    def forward(ctx, x, angle, theta, phi):
        ''' rotate by angle and axis
            Input:
                x: (B,N,3) tensor, here B=1, N=1024
                angle: (B,) tensor, rotate angle
                theta, phi: (B,) tensor, rotation axis decided by theta and phi
                    v = (x,y,z), where x=sin(theta)cos(phi), y=sin(theta)sin(phi), z=cos(theta)
            Return:
                x_rot: (B,N,3) tensor, rotated pointcloud
        '''
        B = x.shape[0]
        cos_alpha = torch.cos(angle)
        sin_alpha = torch.sin(angle)
        ax = torch.sin(theta) * torch.cos(phi) # (B,) tensor, x coordinate of rotation axis
        ay = torch.sin(theta) * torch.sin(phi) # (B,) tensor, y coordinate of rotation axis
        az = torch.cos(theta) # (B,) tensor, z coordinate of rotation axis

        rotation_matrix = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(x.device)

        rotation_matrix[:, 0, 0] = cos_alpha + ax * ax * (1 - cos_alpha)
        rotation_matrix[:, 0, 1] = ax * ay * (1 - cos_alpha) - az * sin_alpha
        rotation_matrix[:, 0, 2] = ax * az * (1 - cos_alpha) + ay * sin_alpha
        rotation_matrix[:, 1, 0] = ax * ay * (1 - cos_alpha) + az * sin_alpha
        rotation_matrix[:, 1, 1] = cos_alpha + ay * ay * (1 - cos_alpha)
        rotation_matrix[:, 1, 2] = ay * az * (1 - cos_alpha) - ax * sin_alpha
        rotation_matrix[:, 2, 0] = ax * az * (1 - cos_alpha) - ay * sin_alpha
        rotation_matrix[:, 2, 1] = ay * az * (1 - cos_alpha) + ax * sin_alpha
        rotation_matrix[:, 2, 2] = cos_alpha + az * az * (1 - cos_alpha)

        x_rotate = torch.matmul(rotation_matrix, x.permute(0, 2, 1)) # (B,3,3) x (B,3,N) -> (B,3,N)
        x_rotate = x_rotate.permute(0, 2, 1) # (B,N,3)
        ctx.save_for_backward(x, angle, theta, phi, rotation_matrix)
        return x_rotate

    @staticmethod
    def backward(ctx, grad_outputs):  # grad_outputs (B,N,3)
        x, angle, theta, phi, rotation_matrix = ctx.saved_tensors
        B = grad_outputs.shape[0]

        cos_alpha, cos_theta, cos_phi = torch.cos(angle), torch.cos(theta), torch.cos(phi)
        sin_alpha, sin_theta, sin_phi = torch.sin(angle), torch.sin(theta), torch.sin(phi)
        ax = sin_theta * cos_phi # (B,)
        ay = sin_theta * sin_phi # (B,)
        az = cos_theta # (B,)
        dax_dtheta, dax_dphi = cos_theta * cos_phi, -sin_theta * sin_phi
        day_dtheta, day_dphi = cos_theta * sin_phi, sin_theta * cos_phi
        daz_dtheta, daz_dphi = -sin_theta, 0


        # d x_rotate / d angle
        dm11_dalpha = (-sin_alpha + ax * ax * sin_alpha).unsqueeze(1) # (B,1)
        dm12_dalpha = (ax * ay * sin_alpha - az * cos_alpha).unsqueeze(1) # (B,1)
        dm13_dalpha = (ax * az * sin_alpha + ay * cos_alpha).unsqueeze(1) # (B,1)

        dm21_dalpha = (ay * ax * sin_alpha + az * cos_alpha).unsqueeze(1) # (B,1)
        dm22_dalpha = (-sin_alpha + ay * ay * sin_alpha).unsqueeze(1) # (B,1)
        dm23_dalpha = (ay * az * sin_alpha - ax * cos_alpha).unsqueeze(1) # (B,1)

        dm31_dalpha = (az * ax * sin_alpha - ay * cos_alpha).unsqueeze(1) # (B,1)
        dm32_dalpha = (az * ay * sin_alpha + ax * cos_alpha).unsqueeze(1) # (B,1)
        dm33_dalpha = (-sin_alpha + az * az * sin_alpha).unsqueeze(1) # (B,1)

        dxrotate_dalpha = torch.cat(
            ((x[:, :, 0] * dm11_dalpha + x[:, :, 1] * dm12_dalpha + x[:, :, 2] * dm13_dalpha).unsqueeze(2), # x[:,:,0] is (B,N), dm11_dalpha is (B,1), after *:(B,N)
             (x[:, :, 0] * dm21_dalpha + x[:, :, 1] * dm22_dalpha + x[:, :, 2] * dm23_dalpha).unsqueeze(2), # (B,N,1)
             (x[:, :, 0] * dm31_dalpha + x[:, :, 1] * dm32_dalpha + x[:, :, 2] * dm33_dalpha).unsqueeze(2)), # (B,N,1)
            dim=2)  # (B,N,3)

        # d x_rotate / d theta
        dm11_dtheta = ((1 - cos_alpha) * 2 * ax * dax_dtheta).unsqueeze(1) # (B,1)
        dm12_dtheta = ((1 - cos_alpha) * (ax * day_dtheta + ay * dax_dtheta) - sin_alpha * daz_dtheta).unsqueeze(1) # (B,1)
        dm13_dtheta = ((1 - cos_alpha) * (ax * daz_dtheta + az * dax_dtheta) + sin_alpha * day_dtheta).unsqueeze(1) # (B,1)

        dm21_dtheta = ((1 - cos_alpha) * (ax * day_dtheta + ay * dax_dtheta) + sin_alpha * daz_dtheta).unsqueeze(1) # (B,1)
        dm22_dtheta = ((1 - cos_alpha) * 2 * ay * day_dtheta).unsqueeze(1) # (B,1)
        dm23_dtheta = ((1 - cos_alpha) * (ay * daz_dtheta + az * day_dtheta) - sin_alpha * dax_dtheta).unsqueeze(1) # (B,1)

        dm31_dtheta = ((1 - cos_alpha) * (ax * daz_dtheta + az * dax_dtheta) - sin_alpha * day_dtheta).unsqueeze(1) # (B,1)
        dm32_dtheta = ((1 - cos_alpha) * (ay * daz_dtheta + az * day_dtheta) + sin_alpha * dax_dtheta).unsqueeze(1) # (B,1)
        dm33_dtheta = ((1 - cos_alpha) * 2 * az * daz_dtheta).unsqueeze(1) # (B,1)

        dxrotate_dtheta = torch.cat(
            ((x[:, :, 0] * dm11_dtheta + x[:, :, 1] * dm12_dtheta + x[:, :, 2] * dm13_dtheta).unsqueeze(2), # (B,N,1)
             (x[:, :, 0] * dm21_dtheta + x[:, :, 1] * dm22_dtheta + x[:, :, 2] * dm23_dtheta).unsqueeze(2), # (B,N,1)
             (x[:, :, 0] * dm31_dtheta + x[:, :, 1] * dm32_dtheta + x[:, :, 2] * dm33_dtheta).unsqueeze(2)), # (B,N,1)
            dim=2)  # (B,N,3)

        # d x_rotate / d phi
        dm11_dphi = ((1 - cos_alpha) * 2 * ax * dax_dphi).unsqueeze(1) # (B,1)
        dm12_dphi = ((1 - cos_alpha) * (ax * day_dphi + ay * dax_dphi) - sin_alpha * daz_dphi).unsqueeze(1) # (B,1)
        dm13_dphi = ((1 - cos_alpha) * (ax * daz_dphi + az * dax_dphi) + sin_alpha * day_dphi).unsqueeze(1) # (B,1)

        dm21_dphi = ((1 - cos_alpha) * (ax * day_dphi + ay * dax_dphi) + sin_alpha * daz_dphi).unsqueeze(1) # (B,1)
        dm22_dphi = ((1 - cos_alpha) * 2 * ay * day_dphi).unsqueeze(1) # (B,1)
        dm23_dphi = ((1 - cos_alpha) * (ay * daz_dphi + az * day_dphi) - sin_alpha * dax_dphi).unsqueeze(1) # (B,1)

        dm31_dphi = ((1 - cos_alpha) * (ax * daz_dphi + az * dax_dphi) - sin_alpha * day_dphi).unsqueeze(1) # (B,1)
        dm32_dphi = ((1 - cos_alpha) * (ay * daz_dphi + az * day_dphi) + sin_alpha * dax_dphi).unsqueeze(1) # (B,1)
        dm33_dphi = ((1 - cos_alpha) * 2 * az * daz_dphi).unsqueeze(1) # (B,1)

        dxrotate_dphi = torch.cat(
            ((x[:, :, 0] * dm11_dphi + x[:, :, 1] * dm12_dphi + x[:, :, 2] * dm13_dphi).unsqueeze(2), # (B,N,1)
             (x[:, :, 0] * dm21_dphi + x[:, :, 1] * dm22_dphi + x[:, :, 2] * dm23_dphi).unsqueeze(2), # (B,N,1)
             (x[:, :, 0] * dm31_dphi + x[:, :, 1] * dm32_dphi + x[:, :, 2] * dm33_dphi).unsqueeze(2)), # (B,N,1)
            dim=2)  # (B,N,3)

        x_grad_input = torch.matmul(grad_outputs, rotation_matrix)  # (B,N,3) x (B,3,3) -> (B,N,3)
        angle_grad_input = torch.matmul(grad_outputs.unsqueeze(2), dxrotate_dalpha.unsqueeze(3)).sum(dim=(1, 2, 3))  # (B,N,1,3) x (B,N,3,1) -> (B,N,1,1)-> sum = (B,)
        theta_grad_input = torch.matmul(grad_outputs.unsqueeze(2), dxrotate_dtheta.unsqueeze(3)).sum(dim=(1, 2, 3))  # (B,N,1,3) x (B,N,3,1) -> (B,N,1,1)-> sum = (B,)
        phi_grad_input = torch.matmul(grad_outputs.unsqueeze(2), dxrotate_dphi.unsqueeze(3)).sum(dim=(1, 2, 3))  # (B,N,1,3) x (B,N,3,1) -> (B,N,1,1)-> sum = (B,)
        # print(x_grad_input.shape, angle_grad_input.shape, theta_grad_input.shape, phi_grad_input.shape)
        return x_grad_input, angle_grad_input, theta_grad_input, phi_grad_input

