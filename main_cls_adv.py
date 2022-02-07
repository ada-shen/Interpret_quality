import os
import argparse
import sys
import time
time_start = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from final_data_train import  ModelNet_Loader, ShapeNetDataset
from models.dgcnn import DGCNN_cls, GCNN_cls
from models.pointnet import PointNetCls
import numpy as np
from torch.utils.data import DataLoader
from tools.final_util import cal_loss, IOStream, set_random
import sklearn.metrics as metrics
import math

from tools.final_util import NUM_POINTS, SHAPENET_CLASS, rot_angle_axis

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_cls_adv.py checkpoints'+'/'+args.exp_name+'/'+'main_cls_adv.py.backup')
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

    if args.model == 'gcnn':
        model = GCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))

    device_ids = [0, 1]
    model = nn.DataParallel(model,device_ids=device_ids)
    model.load_state_dict(torch.load(args.model_path))

    print('************ Adv training ************')
    print('************ start from 100 epoch ************')

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

        for batch, (data, label) in enumerate(train_loader):
            print("batch %d" % batch)
            time0 = time.time()
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]

            ############ gen rot adv sample using BIM ##############
            model.eval()

            alphas = torch.zeros(batch_size, device=device, requires_grad=True)
            thetas = torch.zeros(batch_size, device=device, requires_grad=True)
            phis = torch.zeros(batch_size, device=device, requires_grad=True)
            iterations = args.rotate_adv_iteration
            step = args.rotate_adv_step
            threshold = args.rotate_adv_threshold
            for i in range(iterations):
                data_rot = rot_angle_axis.apply(data, alphas, thetas, phis)  # (B,N,3)
                data_rot = data_rot.permute(0, 2, 1)
                alphas.grad, thetas.grad, phis.grad = None, None, None
                logits = model(data_rot)
                loss = criterion(logits, label)
                loss.backward()
                alpha_grad, theta_grad, phi_grad = alphas.grad.data, thetas.grad.data, phis.grad.data  # (B,), (B,), (B,)
                norm = (theta_grad ** 2 + phi_grad ** 2).sqrt()
                norm[norm == 0] = 1.0

                alphas.data.add_(torch.sign(alpha_grad), alpha=step)  # step
                thetas.data.add_(theta_grad / norm, alpha=step)  # step
                phis.data.add_(phi_grad / norm, alpha=step)  # step
                alphas.data.clamp_(min=-threshold, max=threshold) # only clip rotation angles

            data_rot = rot_angle_axis.apply(data, alphas, thetas, phis).clone().detach() # (B,N,3)
            model.train()
            ############## gen adv sample end ################

            ############ gen rot + trans adv sample using BIM ##############
            model.eval()
            trans = torch.zeros(batch_size, 1, 3, device=device, requires_grad=True)
            iterations = args.trans_adv_iteration
            step = args.trans_adv_step
            threshold = args.trans_adv_threshold
            for i in range(iterations):
                data_trans = data_rot + trans  # (B,N,3) + (B,1,3) -> (B,N,3), broadcast
                data_trans = data_trans.permute(0, 2, 1)
                trans.grad = None
                logits = model(data_trans)
                loss = criterion(logits, label)
                loss.backward()
                trans_grad = trans.grad.data  # (B,1,3)
                trans.data.add_(trans_grad / torch.norm(trans_grad, dim=2, keepdim=True), alpha=step)  # step
                clip_factor = torch.norm(trans.data, dim=2, keepdim=True).clamp_min_(min=threshold)  # (B,1,1)
                trans.data = threshold * trans.data / clip_factor

            data_disturb = (data_rot + trans).clone().detach()

            model.train()
            ############## gen adv sample end ################

            data_rot = data_rot.permute(0,2,1)
            data_disturb = data_disturb.permute(0, 2, 1)  # (B,3,N), rotate + trans adv sample

            opt.zero_grad()

            # rotated pointcloud
            logits = model(data_rot)
            loss = criterion(logits, label)
            loss.backward()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            # rotate + trans pointcloud
            logits = model(data_disturb)
            loss = criterion(logits, label)
            loss.backward()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            opt.step()

            time1 = time.time()
            print("time: ", time1-time0)

        scheduler.step()
        time_epoch_train = time.time()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        print('--------------------------------------------------------------------------------')
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
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
            logits = model(data)
            loss = criterion(logits, label)
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
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        time_epoch_test = time.time()

        ####################
        # Test rot adv
        ####################
        test_adv_loss = 0.0
        count = 0.0
        model.eval()
        test_adv_pred = []
        test_adv_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            # random rotate
            alphas = torch.rand(batch_size, device=device) * math.pi - math.pi/2  # [-pi/2, pi/2]
            thetas = torch.rand(batch_size, device=device) * math.pi # [0, pi)
            phis = torch.rand(batch_size, device=device) * 2 * math.pi # [0, 2pi)

            data = rot_angle_axis.apply(data, alphas, thetas, phis)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_adv_loss += loss.item() * batch_size
            test_adv_true.append(label.cpu().numpy())
            test_adv_pred.append(preds.detach().cpu().numpy())
        test_adv_true = np.concatenate(test_adv_true)
        test_adv_pred = np.concatenate(test_adv_pred)
        test_adv_acc = metrics.accuracy_score(test_adv_true, test_adv_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_adv_true, test_adv_pred)
        outstr = 'Test adv %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_adv_loss * 1.0 / count,
                                                                              test_adv_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        print('training one epoch is %d' %(time_epoch_train-time_epoch))
        print('testing one epoch is %d' %(time_epoch_test-time_epoch_train))

        if epoch % 10 == 9:
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%d.t7' % (args.exp_name, epoch))
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_best.t7' % args.exp_name)

    time_end = time.time()
    outstr='Run Time:%d' %(time_end - time_start)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='gcnn', metavar='N',choices=['gcnn'],help='Model to use, [gcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet10', metavar='N',choices=['modelnet10', 'shapenet'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',choices=['cos', 'step'],help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=20, metavar='N',help='Num of nearest neighbors to use')
    parser.add_argument('--train_rot_y_perturbation', action='store_true', help='Rotation augmentation around y axis.')
    parser.add_argument('--train_rot_all_perturbation', action='store_true', help='Rotation augmentation around 3 axis.')
    parser.add_argument('--drop_point', action='store_true', help='Random drop points to zero.')

    parser.add_argument('--rotate_adv_iteration', type=int, default=7)
    parser.add_argument('--rotate_adv_step', type=float, default=math.pi/10)
    parser.add_argument('--rotate_adv_threshold', type=float, default=math.pi/2)
    parser.add_argument('--trans_adv_iteration', type=int, default=3)
    parser.add_argument('--trans_adv_step', type=float, default=0.4)
    parser.add_argument('--trans_adv_threshold', type=float, default=1.0)
    args = parser.parse_args()

    args.num_points = NUM_POINTS
    if args.train_rot_y_perturbation:
        suffix = "_with_y_rot_da"
    elif args.train_rot_all_perturbation:
        suffix = "_with_all_rot_da"
    else:
        suffix = ""
    args.exp_name = 'exp_MODEL_%s_adv_DATA_%s_POINTNUM_%d_clean%s' % (args.model, args.dataset, args.num_points, suffix)
    args.model_path = "checkpoints/exp_MODEL_%s_DATA_%s_POINTNUM_%d_clean_with_all_rot_da/models/model_99.t7" % (
        args.model, args.dataset, args.num_points) # load normal gcnn from 100 epoch

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

