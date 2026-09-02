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
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset
import scipy.io
import os
from utils import get_dataset, get_model, get_optimizer, get_scheduler
from utils import  LossTracker,run_cmd
from torch.utils.data import DataLoader
from utils import get_pacing_function,balance_order,sample_with_distance, balance_sample_with_distance

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
parser.add_argument('--start_step', default=500, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--topk', default=0, type=int,
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--competency', default=False, action='store_true',
                    help='use competency')
# curriculum params

parser.add_argument('--start_all', default=False, action='store_true',
                    help='use competency')
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
parser.add_argument('--method',default = 'sort')
parser.add_argument('--mixing_step', default = None,type = int)
parser.add_argument('--pre_order',default = None, type = str)
parser.add_argument('--save_mat',default = False, action = 'store_true')
args = parser.parse_args()
val_batchsize = args.batchsize*2
niter = 0
avgpool_outputs= []
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
    print(args.dataset)
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

    criterion_ind = nn.CrossEntropyLoss(reduction="none").to("cuda")
    # initiate a recorder for saving and loading stats and checkpoints
    if  'cscores-orig-order.npz' in args.order_dir:
        temp_path = ''
        if args.score == 'lscore':
            temp_path = os.path.join("orders",args.dataset+'-lscores_train.npz')
        if args.score == 'shfl':
            temp_path = os.path.join("orders",args.dataset+'-shflscores_train.npz')
        if args.score == 'kmeans':
            temp_path = os.path.join("orders",args.dataset+'-kmscores_train.npz')
        if args.score == 'cscore' and args.dataset == 'cifar10':
            temp_path = os.path.join("orders",args.dataset+'-cscores-orig-order.npz')
        if args.score == 'cscore' and (args.dataset == 'cifar10T' or args.dataset == 'cifar100T'):
            temp_path = os.path.join("orders",args.dataset+'-cscores_train.npz')
        if args.score == 'cscore' and args.dataset == 'imagenette':
            temp_path = os.path.join('orders','imagenette_scores.pkl')
        if args.score == 'entropy':
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
            if 'cifar10' == args.dataset:
                url = 'https://pluskid.github.io/structural-regularity/cscores/cifar10-cscores-orig-order.npz'
            #wget.download(url, './orders')
        print(temp_path)
        if args.dataset != 'imagenette':
            temp_x = np.load(temp_path)['scores']
        else:
            with open(temp_path,'rb') as fp:
                temp_dic = pickle.load(fp)
                temp_x = []
                for t in tr_set.paths:
                    #try:
                    temp_x.append(temp_dic[t])
                    #except KeyError:
                    #    temp_x.append(0)
                    
        train_max = np.max(temp_x)
        train_scores = temp_x.copy()
      
        train_ordering = collections.defaultdict(list)
        #diff between train_scores and train_ordering is that latter is not a list
        list(map(lambda a, b: train_ordering[a].append(b), np.arange(len(train_scores)),temp_x))
        train_order = [k for k, v in sorted(train_ordering.items(), key=lambda item: -1*item[1][0])]
     
    else:
        print ('Please check if the files %s in your folder -- orders. See ./orders/README.md for instructions on how to create the folder' %(args.order_dir))
        train_order = [x for x in list(torch.load(os.path.join("orders",args.order_dir)).keys())]

    
    val_scores = []#collections.defaultdict(list)
    temp_path = ''
    if args.score == 'entropy':
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
    elif args.score == 'cscore' and (args.dataset == 'cifar10T' or args.dataset == 'cifar100T'):
        temp_path = os.path.join("orders",args.dataset+'-cscores_val.npz')
    elif args.score == 'cscore' and args.dataset =='imagenette':
            temp_path = os.path.join('orders','imagenette_scores.pkl')
    if args.dataset!='imagenette':
        temp_x = np.load(temp_path)['scores']
    else:
        with open(temp_path,'rb') as fp:
            temp_dic = pickle.load(fp)
            temp_x = []
            for t in val_set.paths:
                #try:
                temp_x.append(temp_dic[t])
                #except KeyError:
                #    temp_x.append(0)
    val_ordering = collections.defaultdict(list)
    val_scores = temp_x#/np.max(temp_x)
    #earlier there was no ordering, scores was called ordering
    list(map(lambda a, b: val_ordering[a].append(b), np.arange(len(temp_x)),temp_x))
    val_order = [k for k, v in sorted(val_ordering.items(), key=lambda item: -1*item[1][0])]
    #list(map(lambda a, b: val_ordering[a].append(b), np.arange(len(temp_x)),temp_x))
    
    val_max = np.max(temp_x)
    data_max = 1#np.max((val_max,train_max))
    val_scores =val_scores/data_max
    train_scores = train_scores/data_max


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
    if args.mixing_step is None:
        args.mixing_step = myiterations+1
    #initial training
    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes), args.half)
    #_ = model.module.avgpool.register_forward_hook(hook)
    _ = model.module.avgpool.register_forward_hook(hook)



    
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=myiterations)

    start_epoch = 0
    total_iter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "iter": [0,], 'tc':[],'tcs':[],'vc':[],'vcs':[],'vce':[],'tce':[],'bsz':[],'bce':[]}
    start_time = time.time()
    

    div = len(train_order)//3
    
    

    trainsets = Subset(tr_set, train_order)
        
    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                                               shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)
    # _train_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,
    #                                            shuffle=False, num_workers=args.workers, pin_memory=True, drop_last = True)
    
    criterion = nn.CrossEntropyLoss().cuda()
    prev_correct_ind = None
    if args.ordering == "standard":
        iterations = 0
        for epoch in range(args.epochs):
 
            train_competency_expected = np.mean(train_scores)
            train_competency_std = np.std(train_scores)
            print('tc',train_competency_expected,train_competency_std, len(train_order))
            
            tr_loss, tr_acc1, iterations,correct_indices = train(train_loader, model, criterion, optimizer,scheduler, epoch,iterations)

            train_competency = np.mean(train_scores[correct_indices])
            val_loss, val_acc1,val_competency, val_competency_std,val_competency_expected, prev_correct_ind = validate(val_loader, model,criterion, val_scores, prev_correct_ind)
            
            #correct_ind = validate(_train_loader, model, criterion)
            if args.save_mat:
                if os.path.exists(args.save_file.replace('.pt','.mat')):
                    _tmp = scipy.io.loadmat(args.save_file.replace('.pt','.mat'))
                else:
                    _tmp = {}
                _tmp[str(niter)+'_ts'] = np.sort(train_scores)
                #_tmp[str(niter)+'_tci'] = correct_ind
                scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)
            niter = niter+1   
            print ("%s epoch %s iterations w/ LEARNING RATE %s"%(epoch, iterations,optimizer.param_groups[0]["lr"]))           
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc1)  
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc1)
            history["iter"].append(iterations)
            history["tce"].append(train_competency_expected)
            history['tc'].append(train_competency)
            history['tcs'].append(train_competency_std)
            history['vc'].append(val_competency)
            history['vcs'].append(val_competency_std)
            history['vce'].append(val_competency_expected)
            torch.save(history,args.save_file)
    else:
        
        
        
        train_order = np.asarray(train_order)
        if not args.start_all:
            B1 = train_order[np.arange(0,div)]
            B2 = train_order[np.arange(div,2*div)]
            B3 = train_order[np.arange(2*div,len(train_order))]
        else:
            B1 = train_order
            B2 = []
            B3 = []


        current_indices = list(B1)
        print ('0 iter data between %s and %s '%(0,len(B1)))
        trainsets = Subset(tr_set, current_indices)
            
            
            
        ts = []
        for o in current_indices:
            ts.append(train_scores[o]/data_max)
        
        
        train_competency_expected = np.mean(ts)
        train_competency_std = np.std(ts)

        print('tce',train_competency_expected, train_competency_std, np.max(ts), np.min(ts))
        
        train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                                                   shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)

        
        dataiter = iter(train_loader)
        step = 0
        next_state = 'B1B2'
        pre_iterations = 0
        box_competency_expected = []
        box_sz = []
        while step < myiterations:   
            tracker = LossTracker(len(train_loader), f'iteration : [{step}]', args.printfreq)
            avgpool_outputs = []
            
            incorrect_indices = []
            correct_indices = []
            for images, target, indices in train_loader:
                
                step += 1
                images, target = cuda_transfer(images, target)
                features = []
                output = model(images)
                #if next_state == 'B1':
                _, predicted = output.max(1)
                incorrect_indices.extend( indices.numpy()[np.where(predicted.cpu().numpy()!=target.cpu().numpy())[0]])
                correct_indices.extend(indices.numpy()[np.where(predicted.cpu().numpy()==target.cpu().numpy())[0]])
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                tracker.update(loss, output, target)
                tracker.display(step-pre_iterations)


            train_competency = 0
            for o in correct_indices:
                train_competency += train_scores[o]/data_max
            train_competency/=len(correct_indices)
            
            train_incompetency = 0
            for o in incorrect_indices:
                train_incompetency += train_scores[o]/data_max
            train_incompetency/=len(incorrect_indices)
            print('Compotent or incompetency',train_competency, train_incompetency, len(incorrect_indices),len(correct_indices))
            #If we hit the end of the dynamic epoch build a new data loader
            pre_iterations = step
            if step>args.start_step:
                if next_state == 'B1B2':
                    if args.topk>0:
                        easy1 = np.intersect1d(B1, train_order[:args.topk])
                    else:
                        easy1 = B1
                        
                    if 'anti' not in args.ordering:
                        move1_2 = np.intersect1d(easy1,incorrect_indices)
                    else:
                        move1_2 = np.intersect1d(easy1,correct_indices)

                    B1 = np.setdiff1d(B1,move1_2).astype('int32')
                    B2 = np.concatenate((B2,move1_2)).astype('int32')
                    
                    current_indices = list(B1)+list(B2)
                    
                    next_state = 'B1B2B3'
                elif next_state == 'B1B2B3':

                    if args.topk>0:
                        easy1 = np.intersect1d(B1, train_order[:args.topk])
                        easy2 = np.intersect1d(B2, train_order[:args.topk])
                        hard2 = np.intersect1d(B2, alt_train_order[:args.topk])
                    else:
                        easy1 = B1
                        hard2 = B2
                        easy2 = B2
                    if 'anti' not in args.ordering:
                        move1_2 = np.intersect1d(easy1,incorrect_indices)
                        move2_1 = np.intersect1d(hard2,correct_indices)
                        move2_3 = np.intersect1d(easy2,incorrect_indices)
                    else:
                        move1_2 = np.intersect1d(easy1,correct_indices)
                        move2_1 = np.intersect1d(hard2,incorrect_indices)
                        move2_3 = np.intersect1d(easy2,correct_indices)
                        
                    B1 = np.concatenate((np.setdiff1d(B1,move1_2),move2_1)).astype('int32')
                    B2 = np.concatenate((np.setdiff1d(B2, np.concatenate((move2_1,move2_3))),move1_2)).astype('int32')
                    B3 = np.concatenate((B3,move2_3)).astype('int32')
                    
                    current_indices = list(B1)+list(B2)+list(B3)
                    
                    next_state = 'B1'
                elif next_state == 'B1':
                    if args.topk>0:
                        easy1 = np.intersect1d(B1, train_order[:args.topk])
                        easy2 = np.intersect1d(B2, train_order[:args.topk])
                        hard2 = np.intersect1d(B2, alt_train_order[:args.topk])
                        hard3 = np.intersect1d(B3, alt_train_order[:args.topk])
                    else:
                        easy1 = B1
                        hard2 = B2
                        easy2 = B2
                        hard3 = B3

                    if 'anti' not in args.ordering:    
                        move1_2 = np.intersect1d(easy1,incorrect_indices)
                        move2_1 = np.intersect1d(hard2,correct_indices)
                        move2_3 = np.intersect1d(easy2,incorrect_indices)
                        move3_1 = np.intersect1d(hard3,correct_indices)
                    else:
                        move1_2 = np.intersect1d(easy1,correct_indices)
                        move2_1 = np.intersect1d(hard2,incorrect_indices)
                        move2_3 = np.intersect1d(easy2,correct_indices)
                        move3_1 = np.intersect1d(hard3,incorrect_indices)

                    B1 = np.concatenate((np.setdiff1d(B1,move1_2),np.concatenate((move2_1,move3_1)))).astype('int32')
                    B2 = np.concatenate((np.setdiff1d(B2, np.concatenate((move2_1,move2_3))),move1_2)).astype('int32')
                    B3 = np.concatenate((np.setdiff1d(B3,move3_1),move2_3)).astype('int32')
                    
                    next_state = 'B1B2'
                    current_indices = list(B1)#list(train_order[B1])
                print("AAA",len(B1), len(B2), len(B3),len(B1)+len(B2)+len(B3))
                print('Next iterations', len(current_indices),next_state)
                trainsets = Subset(tr_set, current_indices)

                train_loader = torch.utils.data.DataLoader(trainsets,\
                                                           batch_size=args.batchsize,\
                                                           shuffle=True, num_workers=args.workers, pin_memory=True, drop_last = True)
                ts = []
                B1_mean = 0
                B2_mean = 0
                B3_mean = 0

                
                for o in current_indices:                        
                    ts.append(train_scores[o]/data_max)

                if len(B1)>0:
                    B1_mean = np.sum(train_scores[B1]/data_max)/len(B1)
                if len(B2)>0:
                
                    B2_mean = np.sum(train_scores[B2]/data_max)/len(B2)
                if len(B3)>0:
                    B3_mean = np.sum(train_scores[B3]/data_max)/len(B3)

                print("Mean competency", B1_mean, B2_mean, B3_mean,len(B1),len(B2),len(B3))
                train_competency_expected = np.mean(ts)

                train_competency_std = np.std(ts)
                box_competency_expected = [B1_mean, B2_mean, B3_mean]
                box_sz = [len(B1), len(B2), len(B3)]
                # start your record
            print ("%s epoch %s iterations w/ LEARNING RATE %s"%(step, step,optimizer.param_groups[0]["lr"]))           
            if step > 50: 
                tr_loss, tr_acc1 = tracker.losses.avg, tracker.top1.avg 

                val_loss, val_acc1,val_competency, val_competency_std, val_competency_expected, prev_correct_ind = validate(val_loader, model,criterion, val_scores, prev_correct_ind)
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
                history['bce'].append(box_competency_expected)
                history['bsz'].append(box_sz)
                torch.save(history,args.save_file)  
                # reinitialization<=================
                model.train()
                #diff = val_competency_wt - train_competency
            if args.save_mat:
                if os.path.exists(args.save_file.replace('.pt','.mat')):
                    _tmp = scipy.io.loadmat(args.save_file.replace('.pt','.mat'))
                else:
                    _tmp = {}
                    _tmp[str(niter)+'_ts'] = np.sort(ts)
                scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)
                                 
                
            print('tce',train_competency_expected,train_competency_std, np.max(ts), np.min(ts))
        
def train(train_loader, model, criterion, optimizer,scheduler, epoch, iterations):
  # switch to train mode
  model.train()
  tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
  pred = []
  tgt = []
  avgpool_outputs = []
  for i, (images, target) in enumerate(train_loader):
    iterations += 1
    images, target = cuda_transfer(images, target)
    output = model(images)
    loss = criterion(output, target)
    _, predicted = output.max(1)
    pred.extend(predicted.cpu().numpy())
    tgt.extend(target.cpu().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tracker.update(loss, output, target)
    tracker.display(i)
    scheduler.step()
  __correct_ind = np.where(np.asarray(pred)==np.asarray(tgt))[0]
  return tracker.losses.avg, tracker.top1.avg,  iterations, __correct_ind




def validate(val_loader, model, criterion,val_scores = None,   prev_correct_ind = None):
  # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        tracker = LossTracker(len(val_loader), f'val', args.printfreq)
        m = 0
        val_competency = 0
        pred = []
        tgt = []
        all_loss = []
        avgpool_outputs.clear()
        #all_outputs = []
        #_all_outputs = []
        for i, (images, target,_) in enumerate(val_loader):
            images, target = cuda_transfer(images, target)
            output = model(images)
            #print(len(avgpool_outputs))
            #_all_outputs.extend(output.cpu().numpy())
            _, predicted = output.max(1)
            correct = predicted.eq(target).cpu().numpy()
            pred.extend(predicted.cpu().numpy())
            tgt.extend(target.cpu().numpy())
            loss = criterion(output, target)
            #_c = nn.CrossEntropyLoss(reduction="none").to("cuda")
            #_loss = _c(output,target)
            #all_loss.extend(_loss.cpu().numpy())
            tracker.update(loss, output, target)
            tracker.display(i)
        #all_outputs.append(_all_outputs)
        correct_ind_org = np.where(np.asarray(pred)==np.asarray(tgt))[0]
        
        
        if val_scores is None:
            return correct_ind
        else:
            val_competency = np.mean(val_scores[correct_ind_org])
            val_competency_std = np.std(val_scores[correct_ind_org])
            val_competency_expected = np.mean(val_scores)

    if args.save_mat:
        if os.path.exists(args.save_file.replace('.pt','.mat')):
            _tmp = scipy.io.loadmat(args.save_file.replace('.pt','.mat'))
            #_tmp = {}
            _tmp[str(niter)+'_vo'] = val_scores.tolist()
            _tmp[str(niter)+'_loss'] = all_loss
            _tmp[str(niter)+'_cio'] = correct_ind_org.tolist()
            scipy.io.savemat(args.save_file.replace('.pt','.mat'),_tmp)

    print('vc',val_competency)
    return tracker.losses.avg, tracker.top1.avg, val_competency,val_competency_std,  val_competency_expected, correct_ind_org

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
    #images = images.to("cuda")
    #target = target.to("cuda")
    if args.half: images = images.half()
    return images, target


if __name__ == '__main__':
    main()

