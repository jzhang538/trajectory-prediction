from __future__ import print_function
import torch
from model import FutureNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest,mse_loss,mse_loss_3dim,kld_loss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import math


## Network Arguments
args = {}
args['use_cuda'] = True
args['in_length'] = 16
args['out_length'] = 25
args['train_flag'] = True
args['continue_flag'] = False



# Initialize network
net = FutureNet(args)
loadEpochs = 0
if args['continue_flag']:
    loadEpochs = 0 #How many epochs has been done
    #net.load_state_dict(torch.load('./trained_models/cslstm_m{}.tar'.format(loadEpochs-1)))
    net.load_state_dict(torch.load('./trained_models/cslstm_m.tar'))
if args['use_cuda']:
    net = net.cuda()


## Initialize optimizer
pretrainEpochs = 40
mseEpochs = 0
nllEpochs = 0
optimizer = torch.optim.Adam(net.parameters())
batch_size = 128
crossEnt = torch.nn.BCELoss()


## Initialize data loaders
trSet = ngsimDataset('./data/Train.mat',args['train_flag'])
valSet = ngsimDataset('./data/Val.mat',args['train_flag'])
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=1,collate_fn=valSet.collate_fn)

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

for epoch_num in range(pretrainEpochs+mseEpochs+nllEpochs-loadEpochs):
    epoch_num +=loadEpochs
    if epoch_num < pretrainEpochs:
        print('MSE')
        stage = 1
    else:
        print('NLL')
        stage = 2


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_pred_loss = 0
    avg_goal_loss = 0
    avg_KL_loss = 0
    avg_recon_loss = 0


    for i, data in enumerate(trDataloader):
        #print(optimizer.param_groups[0]['lr'])
        hist, start_end, nbrs_fut, masks, goal, fut, op_mask = data

        if args['use_cuda']:
            hist = hist.cuda()
            fut = fut.cuda()
            goal = goal.cuda()
            nbrs_fut = nbrs_fut.cuda()
            masks = masks.cuda()
            op_mask = op_mask.cuda()

        goal[:,1] = goal[:,1]/10
  
        pred_fut, hist_recon, z_bag = net(hist,fut,nbrs_fut,masks,goal)
        #print(pred_fut.shape, hist_recon.shape)
        if stage==1:
            pred_fut2 = pred_fut[:,:,:2]
            l_pred = mse_loss_3dim(fut,pred_fut2)
            l_hist = mse_loss_3dim(hist,hist_recon)
            l_goal = mse_loss(fut[-1,:,:],pred_fut2[-1,:,:])
            l_kl = kld_loss(z_bag)
        else:
            l_pred = maskedNLL(pred_fut, fut, op_mask)
            l_kl = kld_loss(z_bag)
            l_goal = mse_loss(fut[-1,:,:],pred_fut2[-1,:,:])
            l_hist = 0

        l = l_pred*1.0 + l_hist*1.0 + l_kl*20.0 + l_goal*1.0
        # Track average train loss and average train time:
        avg_tr_loss += l.item()
        avg_pred_loss += l_pred.item()
        avg_recon_loss += l_hist.item()
        avg_goal_loss += l_goal.item()
        avg_KL_loss += l_kl.item()


        if i%100 == 99:
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),\
           "| Avg pred loss:",format(avg_pred_loss/100,'0.4f'), "| Avg recon loss:",format(avg_recon_loss/100,'0.4f'), \
            "| Avg goal loss:",format(avg_goal_loss/100,'0.4f'), "| Avg kld loss:",format(avg_KL_loss/100,'0.4f'),)
            #train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_pred_loss = 0
            avg_recon_loss = 0
            avg_goal_loss = 0
            avg_KL_loss = 0

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

    if (epoch_num+1)%4 == 0:
        torch.save(net.state_dict(), './trained_models/cslstm_m{}.tar'.format(epoch_num))

torch.save(net.state_dict(), './trained_models/cslstm_m.tar')



