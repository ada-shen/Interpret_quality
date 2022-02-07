import os
import argparse
import sys
import time
time_start = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from final_data_train import ModelNet_Loader, ShapeNetDataset
from models.pointnet import PointNetCls, feature_transform_regularizer
from models.dgcnn import DGCNN_cls, GCNN_cls
from models.pointnet2 import PointNet2ClsMsg
from models.pointconv import PointConvDensityClsSsg
import numpy as np
from torch.utils.data import DataLoader
from tools.final_util import cal_loss, IOStream, set_random
import sklearn.metrics as metrics

from tools.final_util import NUM_POINTS, SHAPENET_CLASS


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls_seed.py checkpoints' + '/' + args.exp_name + '/' + 'main_cls_seed.py.backup')
    os.system('cp models/dgcnn.py checkpoints' + '/' + args.exp_name + '/' + 'dgcnn.py.backup')
    os.system('cp tools/final_util.py checkpoints' + '/' + args.exp_name + '/' + 'final_util.py.backup')
    os.system('cp final_data_train.py checkpoints' + '/' + args.exp_name + '/' + 'final_data_train.py.backup')


def train(args, io):

    if args.dataset == "modelnet10":
        train_loader = DataLoader(ModelNet_Loader(args, partition='train', num_points=args.num_points),
                                  batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(ModelNet_Loader(args, partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "shapenet":
        train_loader = DataLoader(ShapeNetDataset(args, split='train', npoints=args.num_points,
                                                  class_choice=SHAPENET_CLASS, classification=True),
                                  batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(ShapeNetDataset(args, split='test', npoints=args.num_points,
                                                 class_choice=SHAPENET_CLASS, classification=True), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("Dataset does not exist")

    print("#train: ", len(train_loader))
    print("#test: ", len(test_loader))

    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.model == 'pointnet':
        model = PointNetCls(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2ClsMsg(args).to(device)
    elif args.model == 'gcnn':
        model = GCNN_cls(args).to(device)
    elif args.model == 'pointconv':
        model = PointConvDensityClsSsg(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))

    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.model == 'pointnet2' or args.model == 'pointnet':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08) # lr=0.001
    elif args.model == "pointconv":
        print("Use sgd")
        opt = optim.SGD(model.parameters(), lr=args.lr * 10, momentum=args.momentum, weight_decay=1e-4) # lr=0.01
    else:
        print("Use sgd")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4) # lr=0.1

    if args.model == 'dgcnn' or args.model == 'gcnn':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        args.scheduler = 'cos'
    elif args.model == 'pointnet2' or args.model == 'pointnet':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
        args.scheduler = 'step'
    else: # args.model == 'pointconv'
        scheduler = StepLR(opt, step_size=30, gamma=0.7)
        args.scheduler = 'step'

    if args.model == 'pointnet2' or args.model == 'pointnet':
        args.epochs = 200
    elif args.model == 'pointconv':
        args.epochs = 400


    print(args.epochs)

    criterion = cal_loss

    best_test_acc = 0.80
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        time_epoch = time.time()

        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            opt.zero_grad()
            if args.model == 'pointnet':
                logits, trans_feat, _ = model(data)
            else:
                logits = model(data)

            if args.model == "dgcnn" or args.model == "gcnn":
                loss = criterion(logits, label, smoothing=True) # the original paper add label smoothing
            else:
                loss = criterion(logits, label, smoothing=False)

            if args.feature_transform and args.model == 'pointnet':
                loss += feature_transform_regularizer(trans_feat) * args.lambda_ft
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()
        time_epoch_train = time.time()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        print('--------------------------------------------------------------------------------')
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            if args.model == 'pointnet':
                logits, trans_feat, _ = model(data)
            else:
                logits = model(data)

            if args.model == "dgcnn" or args.model == "gcnn":
                loss = criterion(logits, label, smoothing=True)  # the original paper add label smoothing
            else:
                loss = criterion(logits, label, smoothing=False)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        time_epoch_test = time.time()

        print('training one epoch is %d' % (time_epoch_train - time_epoch))
        print('testing one epoch is %d' % (time_epoch_test - time_epoch_train))

        if epoch % 10 == 9:
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (args.exp_name, epoch))

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            outstr = '####################################################################################'
            io.cprint(outstr)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_best.t7' % args.exp_name)

    time_end = time.time()
    outstr = 'Run Time:%d' % (time_end - time_start)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointconv', metavar='N',
                        choices=['pointnet', 'dgcnn', 'gcnn', 'pointnet2', 'pointconv'])
    parser.add_argument('--dataset', type=str, default='modelnet10', metavar='N', choices=['modelnet10','shapenet'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')  # pointnet2 should be 200   pointconv should be 400
    parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform for pointnet")
    parser.add_argument('--lambda_ft', type=float, default=0.001, help="lambda for feature transform for pointnet")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')  # pointnet, pointnet2, pointconv should use step
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--train_rot_y_perturbation', action='store_true',help='Rotation augmentation around y axis.')
    parser.add_argument('--train_rot_all_perturbation', action='store_true',help='Rotation augmentation around 3 axis.')
    parser.add_argument('--drop_point', action='store_true', help='Random drop points to zero.')
    args = parser.parse_args()

    args.num_points = NUM_POINTS
    if args.train_rot_y_perturbation:
        suffix = "_with_y_rot_da"
    elif args.train_rot_all_perturbation:
        suffix = "_with_all_rot_da"
    else:
        suffix = ""
    args.exp_name = 'exp_MODEL_%s_DATA_%s_POINTNUM_%d_clean%s' % (args.model, args.dataset, args.num_points, suffix)

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    set_random(args.seed)

    if args.cuda:
        io.cprint('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    train(args, io)

