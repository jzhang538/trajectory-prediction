from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

#___________________________________________________________________________________________________________________________
# Dataset Description
# 1: Dataset Id
# 2: Vehicle Id
# 3: Frame Number
# 4: Local X
# 5: Local Y
# 6: Lane Id
# 7: Velocity
# 8: Acceleration
# 9: Lateral maneuver
# 10: Longitudinal maneuver
# 11: Num_nbrs
# 12-50: Neighbor Car Ids

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):
    def __init__(self, mat_file, train_flag, t_h=30, t_f=50, d_s=2, step=10, enc_size=32, grid_size=(17,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.step = step # step to calculate clossness
        self.enc_size = enc_size # size of encoder LSTM
        self.flag = train_flag

        self.right_frame = self.D[(self.D[:,10] - 1) == 2]
        self.left_frame = self.D[(self.D[:,10] - 1) == 1]
        self.keep_frame = self.D[(self.D[:,10] - 1) == 0]

        self.ds_c = [self.right_frame,self.left_frame,self.keep_frame]
        print(len(self.ds_c))
        self.grid_size = grid_size
        print("Shape of D:",(self.D).shape)
        print(len(self.left_frame),len(self.right_frame),len(self.keep_frame))


    def __len__(self):
        # length of right frame*3 /epoch
        return len(self.right_frame)*3

    def __getitem__(self, false_idx):
        rand_c = random.randint(0,len(self.ds_c)-1)
        choose_D = self.ds_c[rand_c]
        idx=random.randint(0,len(choose_D)-1)

        dsId = choose_D[idx, 0].astype(int)
        vehId = choose_D[idx, 1].astype(int)
        t = choose_D[idx, 2].astype(int)
        grid = choose_D[idx,12:12+51]
        nbrs_num = choose_D[idx,63].astype(int)

        hist,_ = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,vehId,dsId)

        # nbrs_hist = []
        # nbrs_hist_rel = []
        nbrs_fut = []

        for i in grid:
            if(i!=0):
                # nbr_hist,nbr_hist_rel = self.getHistory(i.astype(int),t,vehId,dsId)
                # nbrs_hist.append(nbr_hist)
                # nbrs_hist_rel.append(nbr_hist_rel)
                nbr_fut = self.getFuture(i.astype(int),t,vehId,dsId)
                nbrs_fut.append(nbr_fut)

        return hist,fut,grid,nbrs_num,nbrs_fut,[dsId,vehId,t]


    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h - self.d_s)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        histPos = vehTrack[stpt:enpt:self.d_s,1:3] - refPos

        hist_prev = histPos[:-1,:]
        hist = histPos[1:,:]
        hist_rel = hist - hist_prev

        if len(hist) < self.t_h//self.d_s + 1:
            return np.empty([0,2]),np.empty([0,2])
        return hist,hist_rel

    # Helper function to get track future
    def getFuture(self, vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3] - refPos

        if len(fut) < self.t_f//self.d_s:
            return np.empty([0,2])
        return fut

    ## Helper function to get track future
    def getFutureRel(self, vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt = np.argwhere(vehTrack[:, 0] == t).item()
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        futPos = vehTrack[stpt:enpt:self.d_s,1:3] - refPos

        fut_prev = futPos[:-1,:]
        fut = futPos[1:,:]
        fut_rel = fut - fut_prev

        if len(fut) < self.t_f//self.d_s:
            return np.empty([0,2])
        return fut_rel

    def getSize(self, vehId, dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        return vehTrack[0,5:7]

    def getCloseness(self,vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt1 = np.argwhere(vehTrack[:, 0] == t).item() + self.step
        enpt1 = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        stpt2 = np.argwhere(refTrack[:, 0] == t).item() + self.step
        enpt2 = np.minimum(len(refTrack), np.argwhere(refTrack[:, 0] == t).item() + self.t_f + 1)

        dis = vehTrack[stpt1:enpt1:self.step,1:3] - refTrack[stpt2:enpt2:self.step,1:3]
        return dis

    def draw_traj(self,vehId,t,dsId):
        plt.xlim((-15,15))
        plt.ylim((-200,400))

        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt = np.argwhere(vehTrack[:, 0] == t).item()
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        refPos = vehTrack[np.where(vehTrack[:,0]==t)][0,1:3]
        traj = vehTrack[stpt:enpt:self.d_s,1:3] - refPos
        print(traj)
        plt.scatter(traj[:,0], traj[:,1], c='red')

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h - self.d_s)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        traj = vehTrack[stpt:enpt:self.d_s,1:3] - refPos
        print(traj)
        plt.scatter(traj[:,0], traj[:,1], c='green')

        plt.legend()
        plt.savefig('./res/draw{} {} {}.jpg'.format(vehId,t,dsId))

    ## Collate function for dataloader
    def collate_fn(self, samples):
        # hist,fut,grid,nbrs_num,nbrs_fut
        # Initialize batch size:
        nbrs_batch_size = 0
        for _,_,_,nbrs_num,_,_ in samples:
            nbrs_batch_size += nbrs_num
        hist_len = self.t_h//self.d_s + 1
        fut_len = self.t_f//self.d_s

        nbrs_fut_batch = torch.zeros(fut_len, nbrs_batch_size,2)
        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(hist_len,len(samples),2)
        fut_batch = torch.zeros(fut_len,len(samples),2)

        start_end_batch = np.zeros([len(samples),2],np.int)
        op_mask_batch = torch.zeros(fut_len,len(samples),2)
        masks_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        masks_batch = masks_batch.byte()
        id_bags_batch = np.zeros([len(samples),3],np.int)

        pos = [0, 0]
        count1=0 #for closeness
        for sampleId,(hist,fut,grid,nbrs_num,nbrs_fut,id_bag) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:hist_len, sampleId, :] = torch.from_numpy(hist[:, :])
            fut_batch[0:fut_len, sampleId, :] = torch.from_numpy(fut[:, :])
            op_mask_batch[0:fut_len,sampleId,:] = 1
            
            start_end_batch[sampleId,0] = count1
            start_end_batch[sampleId,1] = count1+nbrs_num # not include

            id_bags_batch[sampleId,0] = id_bag[0]
            id_bags_batch[sampleId,1] = id_bag[1]
            id_bags_batch[sampleId,2] = id_bag[2]
    
            cnt=0
            for i,nbr in enumerate(grid):
                if nbr!=0:
                    nbrs_fut_batch[:,count1,:] = torch.from_numpy(nbrs_fut[cnt][:,:])

                    pos[0] = i % self.grid_size[0]
                    pos[1] = i // self.grid_size[0]
                    masks_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count1+=1
                    cnt+=1

        return hist_batch, start_end_batch, nbrs_fut_batch, masks_batch, fut_batch, op_mask_batch, id_bags_batch

#________________________________________________________________________________________________________________________________________


## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1):
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                # If we represent likelihood in feet^(-1):
                out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                # If we represent likelihood in m^(-1):
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

def mse_loss(x, x_recon):
    return torch.nn.functional.mse_loss(x_recon, x, reduction='sum')/(x.size(0))

def mse_loss_3dim(x, x_recon):
    return torch.nn.functional.mse_loss(x_recon, x, reduction='sum')/(x.size(0)*x.size(1))

def kld_loss(z):
    mu = z[0]
    logvar = z[1]
    return -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())/mu.size(0)

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
