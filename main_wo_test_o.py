# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import pickle
import time
import warnings
import json
import collections
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset, ConcatDataset
import scipy.io
import os
from utils import get_dataset, get_model, get_optimizer, get_scheduler
from utils import  LossTracker,run_cmd
from torch.utils.data import DataLoader
from utils import get_pacing_function,balance_order,sample_with_distance, balance_sample_with_distance
from temperature_scaling import ModelWithTemperature
from torch.nn.utils import clip_grad_norm_
from eos_utilities import get_hessian_eigenvalues
#from backpack import backpack,extend
#from backpack.extensions import BatchGrad,SumGradSquared,DiagGGNMC,Variance

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data-dir', default='dataset',
                    help='path to dataset')
parser.add_argument('--order-dir', default='cifar10-cscores-orig-order.npz',
                    help='path to train val idx')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: (default: resnet18)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--printfreq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total')
parser.add_argument('--optimizer', default="sgd", type=str,
                    help='optimizer')
parser.add_argument('--scheduler', default="cosine", type=str,
                    help='lr scheduler')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--diff_change', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--diff_low', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--diff_up', default=1, type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--half', default=False, action='store_true',
                    help='training with half precision')
parser.add_argument('--balance', default=False, action='store_true',
                    help='training with half precision')
parser.add_argument('--whiten', default=False, action='store_true',
                    help='training with half precision')
parser.add_argument('--competency', default=False, action='store_true',
                    help='use competency')
# curriculum params
parser.add_argument("--pacing-f", default="linear", type=str, help="which pacing function to take")
parser.add_argument('--pacing-a', default=1., type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--pacing-b', default=1., type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument("--ordering", default="curr", type=str, help="which test case to use. supports: standard, curriculum, anti and random")
parser.add_argument("--score", default="cscore", type=str, help="scorig method")
parser.add_argument("--save_file", default="stat.pt", type=str, help="scorig method")
parser.add_argument('--rand-fraction', default=0., type=float,
                    help='label curruption (default:0)')
parser.add_argument('--avg_validation',default = False, action ='store_true')
parser.add_argument('--cyclic',default = False, action ='store_true')
parser.add_argument('--adap_diff',default = False, action ='store_true')
parser.add_argument('--shift_coord',default = False, action ='store_true')
parser.add_argument('--comp_mult',default = False, action ='store_true')
parser.add_argument('--temp_scaling',default = False, action ='store_true')
parser.add_argument('--loss',default = 'ce', type = str)
parser.add_argument('--track',default = None, type = str)
parser.add_argument('--method',default = 'sort')
parser.add_argument('--mixing_step', default = None,type = int)
parser.add_argument('--pre_order',default = None, type = str)
parser.add_argument('--save_mat',default = False, action = 'store_true')
parser.add_argument('--grad_stats',default = False, action = 'store_true')
args = parser.parse_args()
val_batchsize = args.batchsize*2
niter = 0
avgpool_outputs= []
def get_grad_vector(model):
    """Concatenate all parameter gradients into one vector."""
    grads = []
    for p in model.parameters():
        #print('a',p.grad.shape)
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)
def hutchinson_trace(model, optimizer, loss_fn, X, y, num_samples=1):
    """Estimate trace(Fisher) using Hutchinson's estimator."""
    # Compute log-probabilities
    output = model(X)
    log_probs = F.log_softmax(output, dim=1)

    trace_estimates = []
    for _ in range(num_samples):
        # Draw random ±1 vector z same shape as params
        z = [torch.randint(0, 2, p.shape, device="cuda", dtype=torch.float32) * 2 - 1
             for p in model.parameters() if p.requires_grad]
        #optimizer.zero_grad()

        # Compute vector-Jacobian product (vjp)
        # grad(log p)·z  →  stochastic trace estimate
        total = 0.0
        for i in range(X.size(0)):
            logp = log_probs[i, y[i]]
            grads = torch.autograd.grad(logp, [p for p in model.parameters() if p.requires_grad],
                                        retain_graph=True, create_graph=True)
            total += sum((g*z_i).sum() for g, z_i in zip(grads, z))**2
        trace_estimates.append(total.item() / X.size(0))
    return sum(trace_estimates) / len(trace_estimates)

def hook(module, input, output):
    avgpool_outputs.append(output)
def main():
    global niter
    # features = []
    # def hook(module, input, output):
    #     features.append(output)
    set_seed(args.seed)
    global avgpool_outputs
    
    
    # create training and validation datasets and intiate the dataloaders
    tr_set = get_dataset(args.dataset, args.data_dir, 'train',rand_fraction=args.rand_fraction)
    ho_set = get_dataset(args.dataset, args.data_dir, 'holdout')
    
    print(args.dataset)
    adap_diff = args.adap_diff
    if args.dataset == 'food101':
        train_labels = np.asarray(tr_set._labels)
    else:
        train_labels = np.asarray(tr_set.targets)

    num_classes=len(tr_set.classes)
    
    if args.dataset == "cifar100N":
        val_set = get_dataset("cifar100", args.data_dir, 'val')
        tr_set_clean = get_dataset("cifar100", args.data_dir, 'train')
    elif args.dataset == 'food101':
        val_set = get_dataset(args.dataset, args.data_dir, 'test')
    else:
        val_set = get_dataset(args.dataset, args.data_dir, 'val')
    if args.dataset == 'food101':
        val_labels = np.asarray(val_set._labels)
    else:
        val_labels = np.asarray(val_set.targets)

    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,\
                                               shuffle=True, num_workers=args.workers, pin_memory=True,drop_last = True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batchsize,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    
    criterion_ind = nn.CrossEntropyLoss(reduction="none").cuda()
    # initiate a recorder for saving and loading stats and checkpoints
    if  'cscores-orig-order.npz' in args.order_dir:
        temp_path = ''
        if args.score == 'lscore':
            temp_path = os.path.join("orders",args.dataset+'-lscores_train.npz')
        if args.score == 'shfl':
            temp_path = os.path.join("orders",args.dataset+'-shflscores_train.npz')
        if args.score == 'kmeans':
            temp_path = os.path.join("orders",args.dataset+'-kmscores_train.npz')
        if args.score == 'cluster':
            temp_path = os.path.join("orders",args.dataset+'-clscores_train.npz')
        if args.score == 'proto':
            temp_path = os.path.join("orders",args.dataset+'-proto_sup_train.npz')
        if args.score == 'visc':
            temp_path = os.path.join("orders",args.dataset+'-ic9600_train.npz')
        if args.score == 'svm':
             temp_path = os.path.join("orders",args.dataset+'-svmscores_train.npz')
        if args.score == 'cscore' and args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-cscores-orig-order.npz')
        if args.score == 'cscore' and (args.dataset == 'cifar10T' or args.dataset == 'cifar100T'):
            temp_path = os.path.join("orders",args.dataset+'-cscores_train.npz')
        if args.score == 'cscore' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_consistency_scores.pkl')
        if args.score == 'lscore' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_el2n20_scores.pkl')
        if args.score == 'fscore' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_forgetting_scores.pkl')
        if args.score == 'mscore' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_memo_scores.pkl')
        if args.score == 'entropy' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_bpp_scores_train.pkl')
        elif args.score == 'entropy':
            temp_path = os.path.join("orders",args.dataset+'-ent_train.npz')
        if args.score == 'rateDist':
            temp_path = os.path.join("orders",args.dataset+'-rateDistScores_train.npz')
        if args.score == 'kmsent':
            temp_path = os.path.join("orders",args.dataset+'-kmsent_train.npz')
        if args.score == 'vog':
            temp_path = os.path.join("orders",args.dataset+'-vog_train.npz')
        if not os.path.isfile(temp_path):
            print ('Downloading the data cifar10-cscores-orig-order.npz and cifar100-cscores-orig-order.npz to folder orders')
            if 'cifar100' == args.dataset:
                url = 'https://pluskid.github.io/structural-regularity/cscores/cifar100-cscores-orig-order.npz'
                temp_path = 'orders/cifar100-cscores-orig-order.npz'
            if 'cifar10' == args.dataset:
                url = 'https://pluskid.github.io/structural-regularity/cscores/cifar10-cscores-orig-order.npz'
            #wget.download(url, './orders')
        
        if args.dataset != 'imagenette':
            temp_x = np.load(temp_path)['scores']
        else:
            with open(temp_path,'rb') as fp:
                temp_dic = pickle.load(fp)
                temp_x = []
                for t in tr_set.paths:
                    #try:
                    temp_x.append(temp_dic[t.decode('utf-8')])
                    #except KeyError:
                    #    temp_x.append(0)
                    
        train_max = np.max(temp_x)
        train_scores = temp_x.copy()
        if args.whiten:
            print('whitening per class')
            temp_x = temp_x.astype('float32')
            for cl in range(len(tr_set.classes)):
                _x = np.where(np.asarray(tr_set.targets)==cl)[0]
                #temp_x[_x] = temp_x[_x] - np.mean(temp_x[_x])
                #temp_x[_x] = temp_x[_x]/np.std(temp_x[_x])
                from skimage import exposure
                _l = temp_x[_x]
                _l =_l[:,np.newaxis]
                _l = exposure.equalize_hist(_l)
                _l = np.squeeze(_l)
                temp_x[_x] = _l

        train_ordering = collections.defaultdict(list)
        #diff between train_scores and train_ordering is that latter is not a list
        list(map(lambda a, b: train_ordering[a].append(b), np.arange(len(train_scores)),temp_x))
        train_order = [k for k, v in sorted(train_ordering.items(), key=lambda item: -1*item[1][0])]
        if args.pre_order is not None:
            _t = np.load(args.pre_order)
            if 'cifar10T' in args.dataset:
                train_order = _t[_t<40000]
                print('pre ordered')
        if args.whiten:
           train_ordering = collections.defaultdict(list)
           list(map(lambda a, b: train_ordering[a].append(b), np.arange(len(train_scores)),train_scores))
        #for o in train_order:
        #    print(o,train_scores[o])
    else:
        print ('Please check if the files %s in your folder -- orders. See ./orders/README.md for instructions on how to create the folder' %(args.order_dir))
        train_order = [x for x in list(torch.load(os.path.join("orders",args.order_dir)).keys())]

    
    val_scores = []#collections.defaultdict(list)
    temp_path = ''
    if args.score == 'entropy' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_bpp_scores_val.pkl')
    elif args.score == 'entropy':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-ent_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-ent_val.npz')
    elif args.score == 'rateDist':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-rateDistScores_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-rateDistScores_val.npz')
    elif args.score == 'kmsent':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-kmsent_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-kmsent_val.npz')
    elif args.score == 'lscore':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-lscores_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-lscores_val.npz')
    elif args.score == 'shfl':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-lscores_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-shflscores_val.npz')
    elif args.score == 'vog':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-vog_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-vog_val.npz')
    elif args.score == 'kmeans':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-kmscores_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-kmscores_val.npz')
    elif args.score == 'proto':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-proto_sup_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-proto_sup_val.npz')
    elif args.score == 'visc':
        if args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-ic9600_test.npz')
        else:
            temp_path = os.path.join("orders",args.dataset+'-ic9600_val.npz')
    elif args.score == 'cluster':
        temp_path = os.path.join("orders",args.dataset+'-clscores_val.npz')
    elif args.score == 'svm':
        temp_path = os.path.join("orders",args.dataset+'-svmscores_val.npz')
    elif args.score == 'cscore' and (args.dataset == 'cifar10T' or args.dataset == 'cifar100T'):
        temp_path = os.path.join("orders",args.dataset+'-cscores_val.npz')
    elif args.score == 'cscore' and args.dataset == 'imagenette':
        temp_path = os.path.join('orders','imagenette_consistency_scores.pkl')
    elif args.score == 'lscore' and args.dataset == 'imagenette':
        temp_path = os.path.join('orders','imagenette_el2n20_scores.pkl')
    elif args.score == 'fscore' and args.dataset == 'imagenette':
        temp_path = os.path.join('orders','imagenette_forgetting_scores.pkl')
    elif args.score == 'mscore' and args.dataset == 'imagenette':
        temp_path = os.path.join('orders','imagenette_memo_scores.pkl')        
    if args.dataset!='imagenette':
        if temp_path == '':
            temp_path = 'orders/cifar100-cscores-orig-order.npz'
        temp_x = np.load(temp_path)['scores']
    else:
        with open(temp_path,'rb') as fp:
            temp_dic = pickle.load(fp)
            temp_x = []
            for t in val_set.paths:
                #try:
                temp_x.append(temp_dic[t.decode('utf-8')])
                #except KeyError:
                #    temp_x.append(0)
    val_ordering = collections.defaultdict(list)
    val_scores = temp_x#/np.max(temp_x)
    #earlier there was no ordering, scores was called ordering
    list(map(lambda a, b: val_ordering[a].append(b), np.arange(len(temp_x)),temp_x))
    val_order = [k for k, v in sorted(val_ordering.items(), key=lambda item: -1*item[1][0])]
    #list(map(lambda a, b: val_ordering[a].append(b), np.arange(len(temp_x)),temp_x))
    
    val_max = np.max(temp_x)
    data_max = np.max((val_max,train_max))
    data_max = 1
    val_scores = np.asarray(val_scores)/data_max
    train_scores = np.asarray(train_scores)/data_max
    
 

    train_sorted_scores = []
    train_sorted_order = []
    for o in train_order:
        train_sorted_scores.append(train_ordering[o][0]/data_max)
        train_sorted_order.append(o)
    #print(np.unique(np.asarray(tr_set.targets)[train_order][:3500],return_counts=True))    



    val_sorted_scores = []
    val_sorted_order = []
    for o in val_order:
        val_sorted_scores.append(val_ordering[o][0]/data_max)
        val_sorted_order.append(o)
        
    #val_order = balance_order(val_order, val_set, num_classes=len(tr_set.classes)) 
    alt_train_order = train_order.copy()
    #decide CL, Anti-CL, or random-CL
    if args.ordering == "random":
        np.random.shuffle(train_order)
        np.random.shuffle(val_order)
    elif  args.ordering == "anti_curr" or args.ordering == 'anti_mixed':
        train_order = [x for x in reversed(train_order)]
        val_order = [x for x in reversed(val_order)]
        train_sorted_scores = train_sorted_scores[::-1]
        val_sorted_scores = val_sorted_scores[::-1]
        val_sorted_order  = val_sorted_order[::-1]
        train_sorted_order = train_sorted_order[::-1]
    elif args.ordering == 'mixed':
        alt_train_order = [x for x in reversed(train_order)]

    if args.balance:
        train_order = balance_order(train_order, tr_set, num_classes=len(tr_set.classes)) 
        alt_train_order = balance_order(alt_train_order, tr_set, num_classes=len(tr_set.classes)) 
        print ("check BALANCING",len(train_order),len(tr_set.classes))   
    #print(np.unique(np.asarray(tr_set.targets)[train_order][:3500],return_counts=True))
    
        
    #check the statistics 
    bs = args.batchsize
    N = len(train_order)
    myiterations = (N//bs+1)*args.epochs
    if args.adap_diff:
        track_init_iterations = int((N//bs+1)*args.epochs*0.1)
        track_final_iteratons = int((N//bs+1)*args.epochs*0.9)
    else:
        track_init_iterations = 200
        if args.dataset == 'food101':
            track_init_iterations = 1200
        track_final_iteratons = myiterations
    if args.mixing_step is None:
        args.mixing_step = myiterations+1
    #initial training
    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes), args.half)
    #_ = model.module.avgpool.register_forward_hook(hook)
    #_ = model.module.avgpool.register_forward_hook(hook)

    global net
    if args.avg_validation:
        net = nn.Sequential(nn.Dropout(0.5),
                            model.module.fc)
        #net = model.module.fc
        net.to("cuda")
    
    #model = extend(model,use_converter = True)
    change = False
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=myiterations)
    cycle = 0
    prev_diff = 0
    start_epoch = 0
    total_iter = 0
    tr_loss = 0
    val_loss = 0
    whichway = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "iter": [0,], 'tc':[],'tce':[],'tcs':[],'vc':[],'vce':[],'vcs':[],'vct':[],'vcts':[],'vca':[],'vcas':[],'vcta':[],'vctas':[],'vctec':[],'vcc':[],'track':0,'whichway':[],'gn':[],'fim':[],'gn2':[],'eigs':[],'var_v':[],'fim_v':[],'gn_v':[],'holdout_acc':[],'holdout_loss':[]}
    start_time = time.time()
    trainsets = Subset(tr_set, train_order)
    holdout_subset = Subset(ho_set, np.random.choice(len(ho_set), int(0.25*len(ho_set)), replace=False))
    

    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                                               shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)
    holdout_loader = torch.utils.data.DataLoader(holdout_subset, batch_size=val_batchsize,
                                             shuffle=False, num_workers=args.workers, pin_memory=True,drop_last = True)
    
    # _train_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,
    #                                            shuffle=False, num_workers=args.workers, pin_memory=True, drop_last = True)
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'focal':
        from focal_loss import FocalLoss
        criterion = FocalLoss(gamma=3.0).cuda()
    #criterion = extend(criterion)
    prev_correct_ind = None
    if args.ordering == "standard":
        iterations = 0
        vo = val_order.copy()
        np.random.shuffle(vo)
        rand_val_set = Subset(val_set, list(vo[0:1024]))
        for epoch in range(args.epochs):
 
            train_competency_expected = np.mean(train_scores)
            train_competency_std = np.std(train_scores)
            # train_competency_med = np.median(tc)
            # train_competency_mad = np.median(np.abs(tc-train_competency_med))
            # train_competency_max = np.max(tc)
            # train_competency_min = np.min(tc)
            
            eigs = 0
            #if args.grad_stats and epoch%5==0:
            #    eigs = get_hessian_eigenvalues(model, criterion, rand_val_set, neigs=1,
            #                                   physical_batch_size=64) 
            #    print('eigs', eigs)
            
            tr_loss, tr_acc1, iterations, __correct_ind,grad_norm,fisher_trace = train(train_loader, model, criterion, optimizer,scheduler, epoch,iterations,grad_stats=args.grad_stats)
            train_competency = np.mean(train_scores[__correct_ind])
            print('tce',train_competency_expected,train_competency_std, len(train_order),len(__correct_ind),train_competency)
            testing_ind = list(np.arange(len(val_order)))
            testing_extra_credit_ind = list(np.arange(len(val_order)))
            #val_loss, val_acc1,val_competency, val_competency_std, val_competency_med, val_competency_mad, val_competency_restr, val_competency_calib, val_competency_wt, val_competency_cons, val_competency_cons_std, prev_correct_ind = validate_o(val_loader, model,criterion, val_ordering,train_competency, train_competency_std, train_competency_max, train_competency_min, prev_correct_ind)
            val_loss, val_acc1,val_competency, val_competency_std, val_competency_testing, val_competency_testing_std, val_competency_avg, val_competency_avg_std, val_competency_testing_avg, val_competency_testing_avg_std, val_competency_testing_extra_credit, val_competency_confidence, prev_correct_ind, val_competency_expected,fim_trace_v,grad_norm_v,variance_v = validate(val_loader, model,criterion, val_scores, testing_ind, prev_correct_ind, avg_validation = args.avg_validation, testing_extra_credit_ind = testing_extra_credit_ind, temperature_scaling = args.temp_scaling)
            holdout_loss, holdout_acc1,_ = validate(holdout_loader, model,criterion)
            #correct_ind = validate(_train_loader, model, criterion)
            if args.save_mat:
                if os.path.exists(args.save_file.replace('.pt','.mat')):
                    _tmp = scipy.io.loadmat(args.save_file.replace('.pt','.mat'))
                else:
                    _tmp = {}
                _tmp[str(niter)+'_ts'] = np.sort(train_scores)
                #_tmp[str(niter)+'_tci'] = correct_ind
                scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)
            if  args.grad_stats and (epoch%5==0 or epoch==args.epochs-1):
                state = {
                    'net': model.state_dict(),
                    'acc': val_acc1,
                    'epoch': epoch,
                }
                sdir = './'+args.save_file.replace('.pt','_ckpt')
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                print(sdir)
                torch.save(state, sdir+'/ckpt_'+str(epoch)+'.pth')
            niter = niter+1   
            print ("%s epoch %s iterations w/ LEARNING RATE %s"%(epoch, iterations,optimizer.param_groups[0]["lr"]))           
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc1)  
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc1)
            history["iter"].append(iterations)
            history['tc'].append(train_competency)
            history['tce'].append(train_competency_expected)
            history['tcs'].append(train_competency_std)
            history['vc'].append(val_competency)
            history['vce'].append(val_competency_expected)
            history['vcs'].append(val_competency_std)
            history['vct'].append(val_competency_testing)
            history['vcts'].append(val_competency_testing_std)
            history['vca'].append(val_competency_avg)
            history['vcas'].append(val_competency_avg_std)
            history['vcta'].append(val_competency_testing_avg)
            history['vctas'].append(val_competency_testing_avg_std)
            history['vctec'].append(val_competency_testing_extra_credit)
            history['vcc'].append(val_competency_confidence)
            #history['gn'].append(grad_norm)
            history['eigs'].append(eigs)
            history['gn'].append(grad_norm)
            history['fim'].append(fisher_trace)
            history['fim_v'].append(fim_trace_v)
            history['gn_v'].append(grad_norm_v)
            history['var_v'].append(variance_v)
            history['holdout_acc'].append(holdout_acc1)
            history['holdout_loss'].append(holdout_loss)
            
            #history['fim'].append(fim)
            if change:
                history['track']+=1
            history['whichway'].append(whichway)
            torch.save(history,args.save_file)
    else:
        all_sum = N/(myiterations*(myiterations+1)/2)
        iter_per_epoch = N//bs         
        pre_iterations = 0
        startIter = 0
        startIter_v = 0
        if args.method == 'sort':
            pacing_function = get_pacing_function(myiterations, N, args)
            pacing_function_v = get_pacing_function(myiterations,len(val_order),args)

            startIter_next = pacing_function(0) # <=======================================
            startIter_next_v = pacing_function_v(1)
            print ('0 iter data between %s and %s %s w/ Pacing %s'%(startIter,startIter_next,startIter_next_v, args.pacing_f,))
            trainsets = Subset(tr_set, list(train_order[startIter:max(startIter_next,256)]))
            holdout_subset = Subset(ho_set, list(train_order[startIter:max(startIter_next,256)]))
            if len(holdout_subset) > 0.1*len(ho_set):
                holdout_subset = Subset(holdout_subset, np.random.choice(len(holdout_subset), int((0.25*len(holdout_subset))), replace=False))
                
            vo = val_order.copy()
            np.random.shuffle(vo)
            rand_val_set = Subset(val_set, list(vo[0:1024]))
            testing_ind = list(val_order[startIter_v:max(startIter_next_v,256)])
            testing_extra_credit_ind = list(val_order[startIter_v:max(startIter_next_v+startIter_next_v//10,256)])
            ts = []
            for o in train_order[startIter:max(startIter_next,256)]:
                ts.append(train_ordering[o][0]/data_max)

        elif args.method == 'sample':
            pacing_function = get_pacing_function(myiterations, N//2, args)
            pacing_function_v = get_pacing_function(myiterations,len(val_order)//2,args)
            
            startIter_next = pacing_function(0) # <=======================================
            startIter_next_v = pacing_function_v(1)
            if 2*startIter_next> len(train_scores):
                startIter_next = len(train_scores)//2
            #print("AAA",np.median(train_scores), len(train_scores), train_sorted_scores[20000],np.median(train_sorted_scores) )
            samples = balance_sample_with_distance(train_sorted_order,train_scores, train_labels, num_classes, startIter_next, args.ordering)
            #samples = sample_with_distance(train_sorted_scores,train_scores,  startIter_next, args.ordering)
            
            print("balance",np.unique(np.asarray(tr_set.targets)[np.asarray(samples)],return_counts=True))
            #print(x)
            print ('0 iter data between %s and %s %s w/ Pacing %s'%(startIter,startIter_next,startIter_next_v, args.pacing_f,))
            trainsets = Subset(tr_set, samples)
            ts = train_scores[samples]
            if 2*startIter_next_v> len(val_scores):
                startIter_next_v = len(val_scores)//2
            testing_ind = balance_sample_with_distance(val_sorted_order, val_scores,  val_labels, num_classes, startIter_next_v, args.ordering)
            testing_extra_credit_ind = balance_sample_with_distance(val_sorted_order, val_scores, val_labels, num_classes, startIter_next_v, args.ordering, bw_factor = 1.2)
            #testing_ind = sample_with_distance(val_sorted_scores, val_scores, startIter_next_v, args.ordering)
            #testing_extra_credit_ind = sample_with_distance(val_sorted_scores, val_scores,  startIter_next_v, args.ordering, bw_factor = 1.2)
            
        #ts = train_sorted_scores[startIter:max(startIter_next,256)]
        
        
        train_competency_expected = np.mean(ts)
        train_competency_std = np.std(ts)
        # train_competency_med = np.median(tc)
        # train_competency_mad = np.median(np.abs(tc-train_competency_med))
        # train_competency_max = np.max(tc)
        # train_competency_min = np.min(tc)
        
        # _tmp = {}
        # _tmp[str(niter)+'_ts'] = np.sort(ts)
        # scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)

        print('tce',train_competency_expected, train_competency_std, np.max(ts), np.min(ts),len(ts))
        
        train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                                                   shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)
        holdout_loader = torch.utils.data.DataLoader(holdout_subset, batch_size=val_batchsize,
                                                   shuffle=False, num_workers=args.workers, pin_memory=True)
        
        dataiter = iter(train_loader)
        step = 0
        _step = 0
        iters = 0
        prev_step = 0
        change_step = 0
        while step < myiterations:   
            tracker = LossTracker(len(train_loader), f'iteration : [{step}]', args.printfreq)
            avgpool_outputs = []
            iters +=1
            pred = []
            tgt = []
            total_norm = []

            eigs = 0
            # if args.grad_stats and iters%10==1:
            #     eigs = get_hessian_eigenvalues(model, criterion, rand_val_set, neigs=1,
            #                                    physical_batch_size=64) 
            #     print('eigs', eigs)
            
            grad_norms, fisher_traces = [], []
            print("step iters",step,iters)
            for images, target in train_loader:
                step += 1
                _step +=1
                images, target = cuda_transfer(images, target)
                features = []
                output = model(images)
                _, predicted = output.max(1)
                pred.extend(predicted.cpu().numpy())
                tgt.extend(target.cpu().numpy())   
                loss = criterion(output, target)
                
                optimizer.zero_grad()
                #with backpack(BatchGrad()):
                loss.backward()

                # Calculate the total L2 norm of the gradients
                # The max_norm can be set to a very large value if you only want the norm and not actual clipping.
                # total_grad_norm = clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=2)
                # total_norm.append(total_grad_norm.item())
                # if args.grad_stats  and np.random.rand()>0.5:
                #     grad_vec = get_grad_vector(model)
                #     #print('b',grad_vec.shape)
                #     grad_norm_sq = (grad_vec ** 2).sum().item()
                #     #print(grad_vec.shape)
                    
                #     fisher_trace_est = hutchinson_trace(model, optimizer, criterion, images[:16], target[:16], num_samples=1)

                #     grad_norms.append(grad_norm_sq)
                #     fisher_traces.append(fisher_trace_est)
                
                #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
                optimizer.step()
                scheduler.step()
                tracker.update(loss, output, target)
                tracker.display(step-pre_iterations)
            __correct_ind = np.where(np.asarray(pred)==np.asarray(tgt))[0]
            train_competency = np.mean(np.asarray(ts)[__correct_ind])
            print('tc',len(__correct_ind), train_competency,len(pred),np.mean(ts),np.mean(np.sort(ts)[:len(__correct_ind)]),myiterations,step,N)
            # start your record
            diff = 0
            if step > 50: 
                tr_loss, tr_acc1 = tracker.losses.avg, tracker.top1.avg 
                # val_loss, val_acc1,val_competency, val_competency_std, val_competency_med, val_competency_mad, val_competency_restr, val_competency_calib, val_competency_wt, val_competency_cons, val_competency_cons_std, prev_correct_ind = validate(val_loader, model,criterion, val_ordering,train_competency, train_competency_std, train_competency_max, train_competency_min, prev_correct_ind)

                # _, _,val_competency_ind, val_competency_ind_std, _, _, _, _, _, _, _,_  = validate(val_loader_, model,criterion, np.asarray( val_ordering_),train_competency, train_competency_std, train_competency_max, train_competency_min, None)

                val_loss, val_acc1,val_competency, val_competency_std, val_competency_testing, val_competency_testing_std, val_competency_avg, val_competency_avg_std, val_competency_testing_avg, val_competency_testing_avg_std, val_competency_testing_extra_credit, val_competency_confidence, prev_correct_ind, val_competency_expected,fim_trace_v,grad_norm_v,variance_v = validate(val_loader, model,criterion, val_scores, testing_ind, prev_correct_ind, avg_validation = args.avg_validation, testing_extra_credit_ind = testing_extra_credit_ind, temperature_scaling = args.temp_scaling)
                holdout_loss, holdout_acc1,_ = validate(holdout_loader, model,criterion)
                niter = niter+1             
                # record
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc1)                 
                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc1)  
                history['iter'].append(step)
                history['tce'].append(train_competency_expected)
                history['tc'].append(train_competency)
                history['tcs'].append(train_competency_std)
                history['vc'].append(val_competency)
                history['vce'].append(val_competency_expected)
                history['vcs'].append(val_competency_std)
                history['vct'].append(val_competency_testing)
                history['vcts'].append(val_competency_testing_std)
                history['vca'].append(val_competency_avg)
                history['vcas'].append(val_competency_avg_std)
                history['vcta'].append(val_competency_testing_avg)
                history['vctas'].append(val_competency_testing_avg_std)
                history['vctec'].append(val_competency_testing_extra_credit)
                history['vcc'].append(val_competency_confidence)
                history['holdout_acc'].append(holdout_acc1)
                history['holdout_loss'].append(holdout_loss)
                print('Holdout Train loss', holdout_loss, tr_loss,val_loss)
                #history['gn'].append(np.mean(total_norm))
                # if args.grad_stats:
                #     history['fim'].append(np.mean(fisher_traces))
                #     history['gn'].append(np.mean(grad_norms))
                #     history['eigs'].append(eigs)
                #     history['fim_v'].append(fim_trace_v)
                #     history['gn_v'].append(grad_norm_v)
                #     history['var_v'].append(variance_v)
                # else:
                history['fim'].append(0)
                history['gn'].append(0)
                history['eigs'].append(eigs)
                history['fim_v'].append(0)
                history['gn_v'].append(0)
                history['var_v'].append(0)
                if change:
                    history['track']+=1
                history['whichway'].append(whichway)
                torch.save(history,args.save_file)
                if args.grad_stats and (iters%5==1 ):
                    state = {
                        'net': model.state_dict(),
                        'acc': val_acc1,
                        'epoch': iters,
                    }
                    sdir = './'+args.save_file.replace('.pt','_ckpt')
                    if not os.path.isdir(sdir):
                        os.mkdir(sdir)
                    print(sdir)
                    torch.save(state, sdir+'/ckpt_'+str(iters)+'.pth')
                # reinitialization<=================
                model.train()
                #diff = val_competency_wt - train_competency
            #If we hit the end of the dynamic epoch build a new data loader
            pre_iterations = step
            if args.cyclic and startIter_next>N:
                print("Recycling",startIter_next,N)
                _step = 0
                startIter_next = pacing_function(_step)# <=======================================
                startIter_next_v = pacing_function_v(_step+1)# <=======================================
                # #_tmp = train_order.copy()
                # #train_order = alt_train_order
                # #alt_train_order = _tmp
                # cycle +=1
                # startIter_next = len(train_order)
                # startIter_next_v = len(val_order)
                    
            if (args.method =='sort' and startIter_next <= N) or (args.method =='sample' and startIter_next <N//2):

                gamma = 0.1
                change = False
                whichway = None
                if args.track is not None and step>track_init_iterations:
                    acc_tr_loss = gamma*tr_loss+(1-gamma)*acc_tr_loss
                    acc_val_loss = gamma*val_loss+(1-gamma)*acc_val_loss
                    diff = (acc_val_loss-acc_tr_loss)/acc_tr_loss
                    if args.comp_mult:
                        _acc_tr_loss = acc_tr_loss*val_competency_expected/train_competency_expected
                        diff = (acc_val_loss-_acc_tr_loss)/_acc_tr_loss
                    if adap_diff and diff>0:
                        print('setting diff params, should happen once')
                        args.diff_up = 1.1*diff
                        args.diff_low = 0.9*diff
                        args.diff_change = 0#0.6*(diff-prev_diff)
                        print(args.diff_up,args.diff_low, args.diff_change)
                        adap_diff = False
                    print("current diff ",diff,tr_loss,val_loss)
                    #print('Before change',args.pacing_a, args.pacing_b)
                    if diff>args.diff_up and diff-prev_diff>args.diff_change:
                        if args.track == 'cnst' or args.track == 'aimd':
                            #if args.adap_diff:
                            #    args.pacing_a -= 0.05
                            #else:
                            args.pacing_a -= 0.1 #increasing pace
                        if args.track == 'cnst_sched' or args.track =='aimd_sched':
                            args.pacing_a -= 0.11*(0.91+0.09*step/myiterations) #increasing pace
                        #print('xxxx',args.pacing_a)
                        if args.pacing_a <=0:
                            args.pacing_a = 0.01
                        change = True
                        whichway = 'increasing'
                    if diff <args.diff_low:
                        if args.track == 'cnst':
                            args.pacing_a += 0.1
                        if args.track == 'aimd':
                            args.pacing_a *= 1.1 #slowing OVERFIT
                        if args.track == 'cnst_sched':
                            args.pacing_a += 0.11*(1-0.09*step/myiterations)
                        if args.track == 'aimd_sched':
                            args.pacing_a *= 1.11*(1-0.09*step/myiterations) #slowing OVERFIT
                        change = True

                        if args.pacing_a >=10.0:
                            args.pacing_a = 10.0
                        whichway = 'decreasing'
                    #args.pacing_b = startIter_next/N
                    
                    if change:
                        #_step=1
                        change_step = 0
                        if args.shift_coord:        
                            args.pacing_b = startIter_next/N
                            change_step = prev_step
                            print('shifting coord')
                        pacing_function = get_pacing_function(myiterations-change_step, N, args)
                        pacing_function_v = get_pacing_function(myiterations-change_step, len(val_order), args)
                        print('Tracking changes',whichway,args.pacing_a,args.pacing_b,args.diff_up, args.diff_change,args.diff_low)
                    prev_diff = diff
                else:
                    acc_tr_loss = tr_loss
                    acc_val_loss = val_loss

                    
                if cycle%2 ==0:
                    startIter_next = pacing_function(_step-change_step)# <=======================================
                    startIter_next_v = pacing_function_v(_step+1-change_step)# <=======================================
                else:
                    startIter_next = len(train_order)-pacing_function(_step)
                    startIter_next_v = len(val_order)-pacing_function_v(_step+1)
                    
                if step>=track_final_iteratons and args.adap_diff:
                    print('adap diff, making sure last few iterations are full dataset')
                    startIter_next = len(train_order)
                    startIter_next_v = len(val_order)
                prev_step = step
                print ("%s iter data between %s and %s w/ Pacing %s %s and LEARNING RATE %s "%(step,startIter,startIter_next, startIter_next_v, args.pacing_f, optimizer.param_groups[0]["lr"]))
                if args.method == 'sort':
                    if 'mixed'  not in args.ordering or iters%args.mixing_step:
                        _tr_set = Subset(tr_set, list(train_order[startIter:max(startIter_next,256)]))
                    else:
                        print("mixing ",iters, len(_tr_set), startIter, startIter_next)
                        #_tr_set = ConcatDataset([_tr_set, Subset(tr_set, list(alt_train_order[startIter:max(startIter_next,256)]))])
                        _tr_set = Subset(tr_set, list(alt_train_order[startIter:max(startIter_next,256)]))
                        
                    train_loader = torch.utils.data.DataLoader(_tr_set,\
                                                               batch_size=args.batchsize,\
                                                               shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)
                    testing_ind = list(val_order[startIter_v:max(startIter_next_v,256)])
                    testing_extra_credit_ind = list(val_order[startIter_v:max(startIter_next_v+startIter_next_v//10,256)])
                    
                    if 'mixed'  not in args.ordering or iters%args.mixing_step:
                        ts = []
                        for o in train_order[startIter:max(startIter_next,256)]:                        
                            ts.append(train_ordering[o][0]/data_max)
                    else:
                        ts = []
                        for o in alt_train_order[startIter:max(startIter_next,256)]:                        
                            ts.append(train_ordering[o][0]/data_max)    

                elif args.method == 'sample':
                    if 2*startIter_next> len(train_scores):
                        startIter_next = len(train_scores)//2
                        
                    samples = balance_sample_with_distance(train_sorted_order,train_scores, train_labels, num_classes, startIter_next, args.ordering)
                    #samples = sample_with_distance(train_sorted_scores,train_scores,  startIter_next, args.ordering)
                    print("balance",np.unique(np.asarray(tr_set.targets)[np.asarray(samples)],return_counts=True))
                    train_loader = torch.utils.data.DataLoader(Subset(tr_set, samples),\
                                                                batch_size=args.batchsize,\
                                                                shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)
                    ts = train_scores[samples]
                    if 2*startIter_next_v> len(val_scores):
                        startIter_next_v = len(val_scores)//2
                    testing_ind = balance_sample_with_distance(val_sorted_order, val_scores,  val_labels, num_classes, startIter_next_v, args.ordering)
                    testing_extra_credit_ind = balance_sample_with_distance(val_sorted_order, val_scores, val_labels, num_classes, startIter_next_v, args.ordering, bw_factor = 1.2)
                    #testing_ind = sample_with_distance(val_sorted_scores, val_scores, startIter_next_v, args.ordering)
                    #testing_extra_credit_ind = sample_with_distance(val_sorted_scores, val_scores,  startIter_next_v, args.ordering, bw_factor = 1.2)
                    
                # val_loader_ = torch.utils.data.DataLoader(Subset(val_set, list(val_order[startIter:max(startIter_next_v,256)])),\
                #                                            batch_size=val_batchsize,\
                #                                            shuffle=False, num_workers=args.workers, pin_memory=True)
                
                #ts  = train_scores[startIter:max(startIter_next,256)]
                
                # val_ordering_ = []
                # for o in val_order[startIter:max(startIter_next_v,256)]:
                #     val_ordering_.append(val_ordering[o])
                train_competency_expected = np.mean(ts)
                train_competency_std = np.std(ts)
                if args.save_mat:
                    if os.path.exists(args.save_file.replace('.pt','.mat')):
                        _tmp = scipy.io.loadmat(args.save_file.replace('.pt','.mat'))
                    else:
                        _tmp = {}
                    _tmp[str(niter)+'_ts'] = np.sort(ts)
                    scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)
                                 
                
                print('tce',train_competency_expected,train_competency_std, np.max(ts), np.min(ts),len(ts))
        
def train(train_loader, model, criterion, optimizer,scheduler, epoch, iterations,grad_stats = False):
  # switch to train mode
  model.train()
  tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
  pred = []
  tgt = []
  avgpool_outputs = []
  total_norm = []
  grad_norms, fisher_traces = [], []
  for i, (images, target) in enumerate(train_loader):
    iterations += 1
    images, target = cuda_transfer(images, target)
    #print(np.unique(target.cpu().numpy(),return_counts=True))
    output = model(images)
    loss = criterion(output, target)
    _, predicted = output.max(1)
    pred.extend(predicted.cpu().numpy())
    tgt.extend(target.cpu().numpy())
    optimizer.zero_grad()
    loss.backward()

    # Calculate the total L2 norm of the gradients
    # The max_norm can be set to a very large value if you only want the norm and not actual clipping.
    #total_grad_norm = clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=2)
    #total_norm.append(total_grad_norm.item())
    # if grad_stats and np.random.rand()>0.5:
    #     grad_vec = get_grad_vector(model)
    #     grad_norm_sq = (grad_vec ** 2).sum().item()
    #     fisher_trace_est = hutchinson_trace(model, optimizer, criterion, images[:16], target[:16], num_samples=1)
    
    #     grad_norms.append(grad_norm_sq)
    #     fisher_traces.append(fisher_trace_est)
         
    optimizer.step()
    tracker.update(loss, output, target)
    tracker.display(i)
    scheduler.step()
  correct_ind = np.where(np.asarray(pred)==np.asarray(tgt))[0]
  #if grad_stats:
  #    return tracker.losses.avg, tracker.top1.avg,  iterations, correct_ind, np.mean(grad_norms), np.mean(fisher_traces)
  #else:
  return tracker.losses.avg, tracker.top1.avg,  iterations, correct_ind, 0,0




def validate(val_loader, model, criterion,val_scores = None, testing_ind = None,   prev_correct_ind = None, avg_validation = False, testing_extra_credit_ind = None, temperature_scaling = False,grad_stats = False):
  # switch to evaluate mode
    if temperature_scaling:
        print('Temperature scaling')
        model = ModelWithTemperature(model)
        model.set_temperature(val_loader)
        #T_opt = model.get_temperature()
        
    model.eval()
    # F_kfac = FIM(model=model,
    #              loader=val_loader,
    #              representation=PMatKFAC,
    #              variant='classif_logits')
    # fim = F_kfac.trace()
    # print("FIM",fim)

    with torch.no_grad():


        tracker = LossTracker(len(val_loader), f'val', args.printfreq)
        m = 0
        val_competency = 0
        pred = []
        tgt = []
        all_loss = []
        cfd  = []
        avgpool_outputs.clear()
        #all_outputs = []
        #_all_outputs = []
        variances = []
        fim_trace = []
        grad_norm = []
        for i, (images, target) in enumerate(val_loader):
            images, target = cuda_transfer(images, target)
            output = model(images)
            #print(len(avgpool_outputs))
            #_all_outputs.extend(output.cpu().numpy())
            loss = criterion(output, target)
            output = F.softmax(output, dim=1)
            confidence, predicted = output.max(1)
            correct = predicted.eq(target).cpu().numpy()
            pred.extend(predicted.cpu().numpy())
            tgt.extend(target.cpu().numpy())
            cfd.extend(confidence.detach().cpu().numpy())
            #grad_stats = False
            # if grad_stats:
            #     with backpack(SumGradSquared(),DiagGGNMC()):
            #         loss.backward()
            #     ggn = []
            #     variance = []
            #     sgs = []
            #     for p in model.parameters():
            #             if p.requires_grad:
            #                 sgs.append(p.sum_grad_squared.view(-1))
            #                 ggn.append(p.diag_ggn_mc.view(-1))
            #                 #variance.append(p.variance.view(-1))
            #     ggn = torch.cat(ggn)
            #     variance = torch.cat(variance)
            #     sgs = torch.cat(sgs)
            #     grad_norm.append(sgs.sum().item())
            #     variances.append(0)
            #     fim_trace.append(ggn.sum().item())
                                #print('val',g2.shape,v.shape,var.shape)
            #_c = nn.CrossEntropyLoss(reduction="none").to("cuda")
            #_loss = _c(output,target)
            #all_loss.extend(_loss.cpu().numpy())
            tracker.update(loss, output, target)
            tracker.display(i)
        #all_outputs.append(_all_outputs)
        correct_ind_org = np.where(np.asarray(pred)==np.asarray(tgt))[0]
        
        testing_correct_ind_org = np.where(np.asarray(pred)[testing_ind]==np.asarray(tgt)[testing_ind])[0]
        testing_extra_credit_correct_ind_org = np.where(np.asarray(pred)[testing_extra_credit_ind]==np.asarray(tgt)[testing_extra_credit_ind])[0]
        if avg_validation:

            val_competency_avg_ = []
            val_competency_avg_std_ = []
            val_competency_testing_avg_ = []
            val_competency_testing_avg_std_ = []
            for _ in np.arange(4):
                pred = []
                _all_outputs = []
                for o in avgpool_outputs:
                    o = torch.flatten(o,1)
                    output = net(o)
                    #_all_outputs.extend(output.cpu().numpy())
                    _, predicted = output.max(1)
                    pred.extend(predicted.cpu().numpy())
                
                correct_ind = np.where(np.asarray(pred)==np.asarray(tgt))[0]
                testing_correct_ind = np.where(np.asarray(pred)[testing_ind]==np.asarray(tgt)[testing_ind])[0]
                if val_scores is not None:
                    val_competency_avg_.append(np.mean(val_scores[correct_ind]))
                    val_competency_avg_std_.append(np.std(val_scores[correct_ind]))
                    val_competency_testing_avg_.append(np.mean(val_scores[testing_ind][testing_correct_ind]))
                    val_competency_testing_avg_std_.append(np.std(val_scores[testing_ind][testing_correct_ind]))
                print('CI',len(correct_ind_org), len(correct_ind))
                #all_outputs.append(_all_outputs)

            #pred = np.argmax(np.mean(np.asarray(all_outputs),0),1)
            #aleatoric = np.mean(all_outputs*(1-all_outputs), axis=0)
            #var = np.std(np.asarray(all_outputs),0)
            #ravel_ind = np.ravel_multi_index(np.c_[np.arange(len(pred)),pred].T, var.shape)
            #var = var.flatten()
            #print(np.mean(var[ravel_ind]))
            
            #correct_ind_avg = np.where(np.asarray(pred)==np.asarray(tgt))[0]
            #testing_correct_ind_avg = np.where(np.asarray(pred)[testing_ind]==np.asarray(tgt)[testing_ind])[0]
            #print(np.where(var[ravel_ind][correct_ind_avg]>np.mean(var[ravel_ind]))[0].shape)
            
            #print(np.mean(val_competency_avg_), np.mean(val_competency_avg_std_), np.mean(val_competency_testing_avg_), np.mean(val_competency_testing_avg_std_))
        else:
            correct_ind_avg = correct_ind_org
            testing_correct_ind_avg = testing_correct_ind_org
        if val_scores is None:
            return  tracker.losses.avg, tracker.top1.avg, 0
        else:
            #cfd = np.asarray(cfd)/np.sum(cfd)
            cfd = np.asarray(cfd)
            val_competency = np.mean(val_scores[correct_ind_org])
            val_competency_expected = np.mean(val_scores)
            _jj = np.where(cfd[correct_ind_org]>np.mean(cfd))[0]
            val_competency_cfd = np.mean(val_scores[correct_ind_org[_jj]])#*np.asarray(cfd)[correct_ind_org])
            
            val_competency_std = np.std(val_scores[correct_ind_org])
            val_competency_testing = np.mean(val_scores[testing_ind][testing_correct_ind_org])
            val_competency_testing_std = np.std(val_scores[testing_ind][testing_correct_ind_org])
            val_competency_testing_extra_credit = np.mean(val_scores[testing_extra_credit_ind][testing_extra_credit_correct_ind_org])
            val_competency_confidence = val_competency_cfd#  np.std(val_scores[testing_extra_credit_ind][testing_extra_credit_correct_ind_org])
            if avg_validation:
                val_competency_avg = np.mean(val_competency_avg_)
                val_competency_avg_std = np.mean(val_competency_avg_std_)
                val_competency_testing_avg = np.mean(val_competency_testing_avg_)
                val_competency_testing_avg_std = np.mean(val_competency_testing_avg_std_)
            else:
                val_competency_avg = np.mean(val_scores[correct_ind_avg])
                val_competency_avg_std = np.std(val_scores[correct_ind_avg])
                val_competency_testing_avg = np.mean(val_scores[testing_ind][testing_correct_ind_avg])
                val_competency_testing_avg_std = np.std(val_scores[testing_ind][testing_correct_ind_avg])
            #print(val_competency_avg, val_competency_avg_std, val_competency_testing_avg, val_competency_testing_avg_std)

            # val_competency_cons = val_competency
            # val_competency_cons_std = val_competency_std
            # if prev_correct_ind is not None:
            #     _correct_ind = np.intersect1d(prev_correct_ind, correct_ind)
            #     if len(_correct_ind)>0:
            #         correct_ind = _correct_ind
            #     val_competency_cons = np.mean(val_scores[correct_ind])
            #     val_competency_cons_std = np.mean(val_scores[correct_ind])
    if args.save_mat:
        if os.path.exists(args.save_file.replace('.pt','.mat')):
            _tmp = scipy.io.loadmat(args.save_file.replace('.pt','.mat'))
            #_tmp = {}
            _tmp[str(niter)+'_vo'] = val_scores.tolist()
            _tmp[str(niter)+'_loss'] = all_loss
            _tmp[str(niter)+'_cio'] = correct_ind_org.tolist()
            _tmp[str(niter)+'_cfd'] = cfd
            # _tmp[str(niter)+'_cia'] = correct_ind_avg.tolist()
            # _tmp[str(niter)+'_rci'] = testing_correct_ind.tolist()
            #_tmp[str(niter)+'_ri'] = testing_ind
            scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)

    print('vc',val_competency, val_competency_testing, val_competency_expected)
    #if grad_stats:
    #    return tracker.losses.avg, tracker.top1.avg, val_competency,val_competency_std,  val_competency_testing, val_competency_testing_std, val_competency_avg,  val_competency_avg_std, val_competency_testing_avg, val_competency_testing_avg_std,  val_competency_testing_extra_credit, val_competency_confidence, correct_ind_org, val_competency_expected, np.mean(fim_trace), np.mean(grad_norm), np.mean(variances)
    #else:
    return tracker.losses.avg, tracker.top1.avg, val_competency,val_competency_std,  val_competency_testing, val_competency_testing_std, val_competency_avg,  val_competency_avg_std, val_competency_testing_avg, val_competency_testing_avg_std,  val_competency_testing_extra_credit, val_competency_confidence, correct_ind_org, val_competency_expected, 0,0,0

def set_seed(seed=None):
    if seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    # images = images.to("mps")
    # target = target.to("mps")
    if args.half: images = images.half()
    return images, target


if __name__ == '__main__':
    main()

