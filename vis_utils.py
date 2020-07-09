from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch

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
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, step=10, enc_size = 64):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.step = step # step to calculate clossness
        self.enc_size = enc_size # size of encoder LSTM

        self.right_frame = self.D[(self.D[:,6] - 1) == 2]
        self.left_frame = self.D[(self.D[:,6] - 1) == 1]
        self.keep_frame = self.D[(self.D[:,6] - 1) == 0]

        print("Shape of D:",(self.D).shape)
        print(len(self.left_frame),len(self.right_frame),len(self.keep_frame))


    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2].astype(int)
        nbrs_num = self.D[idx,10].astype(int)
        nbrs_idx = self.D[idx,11:11+nbrs_num]
        pos = self.D[idx,3:5]

        # if(idx==4000):
        #     print(dsId,vehId,t,nbrs_idx,pos)

        hist = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,dsId)

        nbrs_traj = []
        nbrs_fut_traj = []
        nbrs_closeness = []
        for i in nbrs_idx:
            if(i!=-1):
                nbrs_traj.append(self.getHistory(i.astype(int),t,vehId,dsId))
                nbrs_fut_traj.append(self.getNbrsFuture(i.astype(int),t,vehId,dsId))
                nbrs_closeness.append(self.getCloseness(i.astype(int),t,vehId,dsId))
            else:
                nbrs_num-=1

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 9] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 8] - 1)] = 1

        return hist,fut,nbrs_num,nbrs_traj,nbrs_closeness,lat_enc,lon_enc,nbrs_fut_traj


    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
        vehTrack = self.T[dsId-1][vehId-1].transpose()

        if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
             return np.empty([0,2])
        else:
            stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
            enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
            histPos = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
            hisDyn = vehTrack[stpt:enpt:self.d_s,3:5]
            hist = np.concatenate((histPos,hisDyn),axis=1)

        if len(hist) < self.t_h//self.d_s + 1:
            return np.empty([0,2])
        return hist

    ## Helper function to get track history
    def getNbrsFuture(self,vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut


    def getCloseness(self,vehId,t,refVehId,dsId):
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        stpt1 = np.argwhere(vehTrack[:, 0] == t).item() + self.step
        enpt1 = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        stpt2 = np.argwhere(refTrack[:, 0] == t).item() + self.step
        enpt2 = np.minimum(len(refTrack), np.argwhere(refTrack[:, 0] == t).item() + self.t_f + 1)

        dx = refTrack[stpt2:enpt2:self.step,1]-vehTrack[stpt1:enpt1:self.step,1]
        dy = refTrack[stpt2:enpt2:self.step,2]-vehTrack[stpt1:enpt1:self.step,2]
        closeness= np.sqrt(dx*dx+dy*dy)

        # if(vehId==2523 and refVehId==2529 and t==7635):
        #     print(refTrack[stpt2:enpt2:self.step,1])
        #     print(refTrack[stpt2:enpt2:self.step,2])
        #     print(vehTrack[stpt1:enpt1:self.step,1])
        #     print(vehTrack[stpt1:enpt1:self.step,2])
        #     print(closeness)
        return closeness

    def getDSSlidingWindow(self,vehId,t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        
        return 

    ## Collate function for dataloader
    def collate_fn(self, samples):
        # hist,fut,nbrs_num,nbrs_traj,nbrs_closeness,lat_enc,lon_enc
        # Initialize environment length batches:
        env_batch_size = 0
        nbr_batch_size = 0
        for _,_,nbrs_num,_,_,_,_,_ in samples:
            nbr_batch_size += nbrs_num
            env_batch_size += nbrs_num+1 #nbrs and self
        maxlen = self.t_h//self.d_s + 1
        maxlen1 = self.t_f//self.d_s
        env_traj_batch = torch.zeros(maxlen,env_batch_size,4)
        nbrs_fut_batch = torch.zeros(maxlen1,nbr_batch_size,2)
        nbrs_cl_batch = torch.zeros(nbr_batch_size,self.t_f//self.step)

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),4)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        start_end_batch = np.zeros([len(samples),2],np.int)
        nbr_idx_batch = np.zeros([len(samples),2],np.int)

        count=0
        count1=0 #for closeness
        for sampleId,(hist, fut, nbrs_num, nbrs_traj, nbrs_closeness, lat_enc, lon_enc, nbrs_fut) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            # hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            # hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            # hist_batch[0:len(hist),sampleId,2] = torch.from_numpy(hist[:, 2])
            # hist_batch[0:len(hist), sampleId, 3] = torch.from_numpy(hist[:, 3])
            hist_batch[0:len(hist),sampleId,:] = torch.from_numpy(hist[:, :])
            # fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            # fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            fut_batch[0:len(fut), sampleId, :] = torch.from_numpy(fut[:, :])
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            op_mask_batch[0:len(fut),sampleId,:] = 1
            
            start_end_batch[sampleId,0] = count
            start_end_batch[sampleId,1] = count+nbrs_num+1 # not include
            nbr_idx_batch[sampleId,0] = count1
            nbr_idx_batch[sampleId,1] = count1 + nbrs_num

            env_traj_batch[0:maxlen,count,:] = torch.from_numpy(hist[:, :]) #include the history trajectory of ego-vehicle
            count +=1
            # Set up neighbor:
            for i in range(nbrs_num):
                # env_traj_batch[0:maxlen,i,0] = torch.from_numpy(nbrs_traj[i][:,0])
                # env_traj_batch[0:maxlen,i,1] = torch.from_numpy(nbrs_traj[i][:,1])
                # env_traj_batch[0:maxlen,i,2] = torch.from_numpy(nbrs_traj[i][:,2])
                # env_traj_batch[0:maxlen,i,3] = torch.from_numpy(nbrs_traj[i][:,3])
                env_traj_batch[0:maxlen,count,:] = torch.from_numpy(nbrs_traj[i][:,:])
                nbrs_cl_batch[count1,0:] = torch.from_numpy(nbrs_closeness[i][:])
                nbrs_fut_batch[0:maxlen1,count1,:] = torch.from_numpy(nbrs_fut[i][:,:])
                count +=1
                count1 +=1



        return hist_batch, env_traj_batch, nbrs_cl_batch, lat_enc_batch, lon_enc_batch, start_end_batch, fut_batch, op_mask_batch, nbr_idx_batch, nbrs_fut_batch

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

def MSECloseness(y_pred, y_gt):
    out = torch.pow(y_gt-y_pred, 2)
    size = (y_gt.shape[0]*y_gt.shape[1])
    out = torch.sum(out)/size
    return out

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
