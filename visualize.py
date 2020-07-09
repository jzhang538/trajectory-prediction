from __future__ import print_function
import torch
from model import FutureNet
from v_utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt


## Network Arguments
args = {}
args['use_cuda'] = True
args['in_length'] = 16
args['out_length'] = 25
args['train_flag'] = False
args['draw_traj'] = False

# Case for visualization
vis_Id = 0

# Initialize network
net = FutureNet(args)
net.load_state_dict(torch.load('./trained_models/cslstm_m39.tar'))
if args['use_cuda']:
    net = net.cuda()
# Initialize dataset
print("Finish Loading Weights:")
tsSet = ngsimDataset('./data/Test.mat',args['train_flag'])
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=False,num_workers=2,collate_fn=tsSet.collate_fn)

if(args['draw_traj']):
    tsSet.draw_traj(2725,8535,5)
else:
    print("Begin Searching:")
    for i, data in enumerate(tsDataloader):
        if(vis_Id>=(i*128+128)):
            continue

        hist, start_end, nbrs_fut, masks, fut, op_mask, id_bags = data

        if args['use_cuda']:
            hist = hist.cuda()
            fut = fut.cuda()
            nbrs_fut = nbrs_fut.cuda()
            masks = masks.cuda()
            op_mask = op_mask.cuda()


        if(vis_Id<(i*128+128)):
            temp_Id = vis_Id%128
            print("fut:",fut[:,temp_Id,:])
            goal = fut[-1,:,:].squeeze()
            # goal[temp_Id,0] = 0
            # goal[temp_Id,1] = 100
            goal[:,1] = goal[:,1]/10
            #pred_fut, hist_recon, z_bag = net(hist,fut,nbrs_fut,masks,goal)
            pred_fut, hist_recon = net.inference(hist,fut,nbrs_fut,masks,goal)

            plt.xlim((-15,15))
            plt.ylim((-200,400))
            
            c_list=['orange','yellow','blue','purple','grey','black','brown','pink','lightblue','lightgreen','cyan','darkred','darkorange','darkblue','darkgreen','tan']

            nbrs_idx1 = int(start_end[temp_Id,0])
            nbrs_idx2 = int(start_end[temp_Id,1])
            ct=0
            for j in range(nbrs_idx1,nbrs_idx2):
                nbr_traj=nbrs_fut[:,j,:].cpu()
                plt.scatter(nbr_traj[:,0], nbr_traj[:,1], c=c_list[ct])
                ct+=1

            traj=hist[:,temp_Id,:].cpu()
            plt.scatter(traj[:,0], traj[:,1], c='green')
            traj=hist_recon[:,temp_Id,:].detach().cpu()
            plt.scatter(traj[:,0], traj[:,1], c='red')

            traj=fut[:,temp_Id,:].cpu()
            plt.scatter(traj[:,0], traj[:,1], c='green')
            print("fut:",traj)
            traj=pred_fut[:,temp_Id,:].detach().cpu()
            plt.scatter(traj[:,0], traj[:,1], c='red')
            print("pred_fut:",traj)

            plt.legend()
            print(id_bags[temp_Id])
            plt.savefig('./res/{} {} {}.jpg'.format(id_bags[temp_Id,0], id_bags[temp_Id,1], id_bags[temp_Id,2]))
            break



