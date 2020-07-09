from __future__ import print_function
import torch
from model import FutureNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt


## Network Arguments
args = {}
args['use_cuda'] = True
args['in_length'] = 16
args['out_length'] = 25
args['train_flag'] = True
args['continue_flag'] = False


# Evaluation metric:
metric = 'rmse'  #or rmse

# Initialize network
net = FutureNet(args)
net.load_state_dict(torch.load('./trained_models/cslstm_m39.tar'))
if args['use_cuda']:
    net = net.cuda()
# Initialize dataset
print("Finish Loading Weights:")
tsSet = ngsimDataset('./data/Test.mat',args['train_flag'])
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=False,num_workers=8,collate_fn=tsSet.collate_fn)

counts = 0
lossVals = 0
print("Begin Searching:")
for i, data in enumerate(tsDataloader):
    hist, start_end, nbrs_fut, masks, goal, fut, op_mask = data

    if args['use_cuda']:
        hist = hist.cuda()
        fut = fut.cuda()
        nbrs_fut = nbrs_fut.cuda()
        masks = masks.cuda()
        goal = goal.cuda()
        op_mask = op_mask.cuda()

    goal[:,1] = goal[:,1]/10
    #pred_fut, hist_recon, z_bag = net(hist,fut,nbrs_fut,masks,goal)
    pred_fut, hist_recon = net.inference(hist,fut,nbrs_fut,masks,goal)

    if metric == 'nll':
        l, c = maskedNLLTest(pred_fut, 0, 0, fut, op_mask,use_maneuvers=False)
    else:
        # Forward pass
        l, c = maskedMSETest(pred_fut, fut, op_mask)

    lossVals +=l.detach()
    counts += c.detach()
    if  ((i*128)%1280)==0:
        print(i*128)

if metric == 'nll':
    print(lossVals / counts)
else:
    print(torch.pow(lossVals / counts,0.5)*0.3048)


