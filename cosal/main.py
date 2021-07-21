from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as fn
# import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

import random

from cosal import ppr
from cosal.cosal import COSAL
from cosal.dataset import PPRDataset
from cosal.utils import *

import argparse
import time
import uuid

# args parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--data_file', type=str, default='cora')
parser.add_argument('--topk', type=int, default=10,
                    help='Number of PPR neighbors for each node.')
parser.add_argument('--alpha', type=float, default=0.25,
                    help="Teleport probability when computing ppr matrix.")
parser.add_argument('--eps', type=float, default=1e-3,
                    help="Stopping threshold for ACL's ApproximatePR.")
parser.add_argument('--isReg', action='store_true', default=True,########----0309 True 2 False
                    help="Aspect regularization flag.")######
parser.add_argument('--reg_coef', type=float, default=0.00012,
                    help="Aspect regularization strength.")#help="\lambda in Eq.15"
parser.add_argument('--threshold', type=float, default=0.1,
                    help="Aspect regularization standard.")#help="\epsilon in Eq.13"

parser.add_argument('--init_mtd', type=int, default=0,
                    help='The initlization way of sub-representation.')
# init_mtd == 0: initialze u as itself, i.e., z_(u,k),
# init_mtd == 1: initialize u as mean of neighbors' feature
# init_mtd == 2: initialize u as sum of neighbors' feature weighted by ppr

parser.add_argument('--attention', type=int, default=1,
                    help='The method to compute attention.')
# attention == 0: #attention = softmax(feature_sim * ppr)
# attention == 1: #attention = softmax( beta * feature_sim + (1-beta) * softmax(ppr) )
# attention == 2: #attention = feature_sim
parser.add_argument('--beta', type=float, default=0.83,
                    help='The trad-off weight between feature similarity and ppr similarity when choosing the attention method 1.')

parser.add_argument('--isSL', action='store_true', default=True,
                    help="Self-loop adjacent matrix.")

# parser.add_argument('--device', type=int, default=0,
#                     help="The GPU ID chosed to use.")

parser.add_argument('--hidden_size', type=int, default=128,
                    help='Number of hidden units per capsule.')
parser.add_argument('--cap_hidden_size', type=int, default=32,
                    help='Maximum number of capsules per layer.')
parser.add_argument('--rouit', type=int, default=2,
                    help='Number of iterations when routing.')

parser.add_argument('--lr', type=float, default=5e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--max_epochs', type=int, default=200,
                    help='Max number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512,
                    help="atch size for training.")     
# parser.add_argument('--batch_mult_val', type=int , default=4 , help='Multiplier for validation batch size')


parser.add_argument('--eval_step', type=int , default=20 , help='Accuracy is evaluated after every this number of steps')
# parser.add_argument('--run_val', action='store_true', default=False, help='Evaluate accuracy on validation set during training')########----0309

parser.add_argument('--early_stop', action='store_true', default=False,
                    help='Use early stopping.')
parser.add_argument('--patience', type=int , default=50 , help='Patience for early stopping')

#parser.add_argument('--', type= , default= , help='')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='Insist on using CPU instead of CUDA.')#if you want to use gpu, then not list "--cpu" when you run this code

#-----0917

parser.add_argument('--is_xNorm', action='store_true', default=True,
                    help="Feature matrix normalization flag.")#0107######
parser.add_argument('--is_pprNorm', action='store_true', default=False,
                    help="PPR value matrix normalization flag.")
parser.add_argument('--isSamp', action='store_true', default=False,
                    help="Weighted sampling flag.")#0107########
parser.add_argument('--split_seed', type=int, default=0, help='Seed for splitting the dataset into train/val/test.')
parser.add_argument('--ntrain_div_classes', type=int, default=20,
                    help='Number of training nodes divided by number of classes.')
#     parser.add_argument('--attr_normalization', type=str , default=None , help='Attribute normalization. Not used in the paper.')

parser.add_argument('--ppr_normalization', type=str , default='sym' , help='Adjacency matrix normalization for weighting neighbors.')
#-----0917          
parser.add_argument('--seed', type=int, default='8899', help='random seed')

# if args_str is None:
args = parser.parse_args()
# else:
#     args = parser.parse_args(args_str.split())

# Load Data
data_dir = args.data_dir 
data_file = args.data_file
start = time.time()
if(data_file == 'cora' or data_file == 'citeseer' or data_file == 'pubmed'):
    (adj_matrix, attr_matrix, labels, train_idx, val_idx, test_idx) = get_data_v2(data_file, data_dir)     
    checkpt_file = './pretrained/'+ uuid.uuid4().hex+'.pt'
elif('ogbn' in data_file):
    data_dir = "../dataset_ogb"
    (adj_matrix, attr_matrix, labels, train_idx, val_idx, test_idx, nc) = get_data_ogb_v2(data_dir, data_file, args.split_seed, args.ntrain_div_classes)
    checkpt_file = './pretrained/'+ uuid.uuid4().hex+'.pt'
else:
    (adj_matrix, attr_matrix, labels, train_idx, val_idx, test_idx) = get_data(data_dir+f"{args.data_file}",seed=args.split_seed,ntrain_div_classes=args.ntrain_div_classes) #--20210311
    checkpt_file = './pretrained/'+ uuid.uuid4().hex+'.pt'
try:
    d = attr_matrix.n_columns
except AttributeError:
    d = attr_matrix.shape[1]
if('ogbn' in data_file):
    pass
else:
    nc = labels.max() + 1

# set random seed for reproduciable
# func for setting random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
    return 0
seed = args.seed
set_random_seed(seed)

# hyper-paramters space
def run_batch(model, xbs, yb, optimizer, isReg, ncaps, threshold, reg_coef, train):
    # Set model to training mode
    if train:
        model.train()
    else:
        model.eval()
    if train:
        optimizer.zero_grad()
    t1 = time.time()
    with torch.set_grad_enabled(train):
        if isReg and train:
            prob, x_reshaped = model(*xbs) 
            loss = fn.nll_loss(prob, yb)
            if ncaps != 1:
                div_metric = 0.0
                for i in range(ncaps):
                    for j in range(i+1,ncaps):
                        sim_matrix = fn.cosine_similarity(x_reshaped[:, i, :], x_reshaped[:, j, :])
                        mask = torch.abs(sim_matrix) > threshold
                        div_metric += (torch.abs(torch.masked_select(sim_matrix, mask))).sum()
                div_reg = reg_coef * div_metric
                loss += div_reg   
        else:
            prob = model(*xbs)
            loss = fn.nll_loss(prob, yb)
        if train:
            loss.backward()
            optimizer.step()
    batch_time = time.time() - t1
    return loss, prob, batch_time

# training function
def train_func(args, adj_matrix, attr_matrix, labels, train_idx, val_idx, d, nc, checkpt_file):
    #init parameter
    eval_step = args.eval_step
    alpha, eps, topk, isSL = args.alpha, args.eps, args.topk, args.isSL
    is_pprNorm, isSamp, ppr_normalization = args.is_pprNorm, args.isSamp, args.ppr_normalization
    isReg, threshold, reg_coef =  args.isReg, args.threshold, args.reg_coef
    init_mtd, attention, beta =  args.init_mtd, args.attention, args.beta
    hidden_size, cap_hidden_size, rouit =  args.hidden_size, args.cap_hidden_size, args.rouit
    ncaps = int(hidden_size/cap_hidden_size)
    lr, weight_decay, dropout =  args.lr, args.weight_decay, args.dropout
    max_epochs, batch_size, eval_step =  args.max_epochs, args.batch_size, args.eval_step
    seed =  args.seed
    is_xNorm =  args.is_xNorm
#     d, nc =  
    cpu = args.cpu
    
    set_random_seed(seed)
    
    #model
    model = COSAL(d, nc, hidden_size, ncaps, init_mtd, attention, beta, dropout, rouit, isReg, cpu, is_xNorm)
    
    #device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    torch.cuda.set_device(device)
    model.to(device)
    
    # optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    #dataloader
    t2 = time.time()
    topk_train = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk, isSL,
                                     isSamp, is_pprNorm, normalization=ppr_normalization)
    train_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_train, indices=train_idx, labels_all=labels)
    preprocessing_time = time.time() - t2
    
    topk_val = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk, isSL,
                                   isSamp, is_pprNorm, normalization=ppr_normalization)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, 
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(train_set),
            batch_size=batch_size, drop_last=False
        ),
        batch_size=None,
        num_workers=0
    )
    
    val_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_val, indices=val_idx, labels_all=labels)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, 
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(val_set),
            batch_size=batch_size, drop_last=False
        ),
        batch_size=None,
        num_workers=0
    )
    #epoch
    best_val_acc = 0.0
    epoch_step = 0
    train_time = 0
    for epoch in range(max_epochs):
        running_loss = 0.0
        train_num = 0
        train_correct = 0
        for xbs, yb in train_loader:# ---这里调用了__getitem__()
            xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)
            loss_batch, prob_batch, batch_time = run_batch(model, xbs, yb, optimizer, isReg, ncaps, threshold, reg_coef, train=True)
            train_time += batch_time
            running_loss += loss_batch.item()
            train_num += yb.size(0)
            _, train_predicted = torch.max(prob_batch.data, 1)
            train_correct += (train_predicted==yb).sum().item()
            
            if epoch_step % eval_step == 0:#evaluation
                val_loss = 0.0
                val_steps = 0
                val_num = 0
                val_correct = 0
                for xbs, yb in val_loader:
                    xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)
                    loss_batch, prob_batch, _ = run_batch(model, xbs, yb, optimizer, isReg, ncaps, threshold, reg_coef, train=False)
                    val_loss += loss_batch.item()
                    _, predicted = torch.max(prob_batch.data, 1)
                    val_num += yb.size(0)
                    val_correct += (predicted==yb).sum().item()
                    val_steps += 1
                val_acc = val_correct / val_num
                train_acc = train_correct / train_num
                print(f"Epoch {epoch}, step {epoch_step}: train loss {running_loss/train_num:.5f}, val loss {val_loss/val_num:.5f}, train acc {train_acc:.5f}, val acc {val_acc:.5f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), checkpt_file)
            epoch_step += 1
    tr_time = preprocessing_time + train_time
    return tr_time
#     model.load_state_dict(torch.load(checkpt_file))
#     torch.save(model.state_dict(), checkpt_file)
    
# train
tr_time = train_func(args, adj_matrix, attr_matrix, labels, train_idx, val_idx, d, nc, checkpt_file)

# create the best trained model
ncaps = int(args.hidden_size/args.cap_hidden_size)
best_trained_model = COSAL(d, nc, args.hidden_size, ncaps, args.init_mtd, args.attention, args.beta, args.dropout, args.rouit, args.isReg, args.cpu, args.is_xNorm)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
torch.cuda.set_device(device)
best_trained_model.to(device)
best_trained_model.load_state_dict(torch.load(checkpt_file))

# test_accuracy func
def test_accuracy(model, test_set, device, batch_size, seed):
    set_random_seed(seed)
    testloader = torch.utils.data.DataLoader(
        dataset=test_set, 
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(test_set),
            batch_size=batch_size, drop_last=False
        ),
        batch_size=None,
        num_workers=0
    )
    correct = 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for xbs, yb in testloader:
            xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)
            #predict
            prob = model(*xbs)
            _, predictions = torch.max(prob.data, 1)
            total += yb.size(0)
            correct += (predictions==yb).sum().item()
    return correct / total

# test on the best trained model
topk_test = ppr.topk_ppr_matrix(adj_matrix, args.alpha, args.eps, test_idx, args.topk, args.isSL,
                                args.isSamp, args.is_pprNorm, normalization=args.ppr_normalization)
test_set = PPRDataset(attr_matrix_all=attr_matrix, ppr_matrix=topk_test, indices=test_idx, labels_all=labels)
test_acc = test_accuracy(best_trained_model, test_set, device, batch_size=10000, seed=seed)
print("Best trial test set accuracy: {}".format(test_acc))
print("training time: {:.4f}s".format(tr_time))
