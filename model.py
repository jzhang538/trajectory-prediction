from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F
import numpy as np

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for (start, end) in seq_start_end:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class FutureNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(FutureNet, self).__init__()

        ## Unpack arguments
        self.args = args
        ## Use gpu flag
        self.use_cuda = args['use_cuda']
        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']
        ## Sizes of network layers
        self.in_length = args['in_length']
        self.out_length = args['out_length']


        self.traj_embedding_size = 16
        self.encoder_size = 32

        self.soc_conv_depth = 64
        self.conv_3x1_depth = 16
        self.grid_size=[17,3]
        self.soc_embedding_size = (((self.grid_size[0]-4)+1)//2)*self.conv_3x1_depth

        self.latent_size = 64
        self.decoder_input_dim = 32
        self.decoder_output_dim= 32
        
        self.goal_dim=2
        self.goal_embedding_size=16
        # self.n_units = ([self.env_encoder_size+self.pos_size] + [self.graph_network_hidden_dims] + [self.graph_network_out_dims])
        # self.n_heads=[4,1]
        # self.dropout=0
        # self.alpha=0.2
        
        # Future Encode
        self.traj_emb = torch.nn.Linear(2,self.traj_embedding_size)
        self.fut_enc_lstm = torch.nn.LSTM(self.traj_embedding_size,self.encoder_size,1)

        #Hist Encode
        self.hist_enc_lstm = torch.nn.LSTM(self.traj_embedding_size,self.encoder_size,1)
        self.op_hist = torch.nn.Linear(self.encoder_size,2)

        # Social Pooling
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # Latent Layer
        self.goal_emb = torch.nn.Linear(self.goal_dim,self.goal_embedding_size)
        self.latent_enc = torch.nn.Linear(self.encoder_size + self.soc_embedding_size + self.goal_embedding_size,self.latent_size)
        self.conditional_mean = torch.nn.Linear(self.latent_size,self.latent_size)
        self.conditional_logvar = torch.nn.Linear(self.latent_size,self.latent_size)
        self.latent_dec = torch.nn.Linear(self.latent_size + self.soc_embedding_size + self.goal_embedding_size,self.decoder_input_dim)

        # Future Decode
        self.fut_dec_lstm = torch.nn.LSTM(self.decoder_input_dim,self.decoder_output_dim)
        self.op = torch.nn.Linear(self.decoder_output_dim,5)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    ## Forward Pass
    def forward(self,hist,fut,nbrs_fut,masks,goal):
        ego_hist_enc,(ego_hist_h,ego_hist_c)=self.hist_enc_lstm(self.leaky_relu(self.traj_emb(hist)))
        #ego_hist_enc = ego_hist_enc.view(ego_hist_enc.shape[1],ego_hist_enc.shape[2])
        hist_recon = self.op_hist(ego_hist_enc)

        # Forward pass environment:
        _,(nbrs_fut_enc,_) = self.fut_enc_lstm(self.leaky_relu(self.traj_emb(nbrs_fut)))
        nbrs_fut_enc = nbrs_fut_enc.view(nbrs_fut_enc.shape[1], nbrs_fut_enc.shape[2])
        fut_soc_enc = torch.zeros_like(masks).float()
        fut_soc_enc = fut_soc_enc.masked_scatter_(masks, nbrs_fut_enc)
        fut_soc_enc = fut_soc_enc.permute(0,3,2,1)
        fut_soc_enc = self.leaky_relu(self.soc_conv(fut_soc_enc))
        # [num_nbrs, C, 15, 1]
        # Add Nonlocal
        fut_soc_enc = self.leaky_relu(self.conv_3x1(fut_soc_enc))
        # [num_nbrs, C, 13, 1]
        fut_soc_enc = self.soc_maxpool(fut_soc_enc)
        fut_soc_enc = fut_soc_enc.view(-1,self.soc_embedding_size)

        goal = self.goal_emb(goal)
        cond = torch.cat([fut_soc_enc,goal],dim=1)
        cond = cond.repeat(self.out_length,1,1)

        ego_fut_enc,_ = self.fut_enc_lstm(self.leaky_relu(self.traj_emb(fut)),[ego_hist_h,ego_hist_c])
        #latent enc
        enc = torch.cat([ego_fut_enc,cond],dim=2)
        enc = self.leaky_relu(self.latent_enc(enc))
        z_mean,z_std = self.condition_forward_(enc)
        #print(z_mean.shape,z_std.shape)
        #latent dec
        enc_ = self.reparametrize(z_mean,z_std)
        enc_ = torch.cat([enc_,cond],dim=2)
        enc_ = self.leaky_relu(self.latent_dec(enc_))

        h_dec,_ = self.fut_dec_lstm(enc_,[ego_hist_h,ego_hist_c])
        fut_pred = self.op(h_dec)
        fut_pred = outputActivation(fut_pred)

        return fut_pred,hist_recon,[z_mean,z_std]

    def inference(self,hist,fut,nbrs_fut,masks,goal):
        ego_hist_enc,(ego_hist_h,ego_hist_c)=self.hist_enc_lstm(self.leaky_relu(self.traj_emb(hist)))
        hist_recon = self.op_hist(ego_hist_enc)

        # Forward pass environment:
        _,(nbrs_fut_enc,_) = self.fut_enc_lstm(self.leaky_relu(self.traj_emb(nbrs_fut)))
        nbrs_fut_enc = nbrs_fut_enc.view(nbrs_fut_enc.shape[1], nbrs_fut_enc.shape[2])
        fut_soc_enc = torch.zeros_like(masks).float()
        fut_soc_enc = fut_soc_enc.masked_scatter_(masks, nbrs_fut_enc)
        fut_soc_enc = fut_soc_enc.permute(0,3,2,1)
        fut_soc_enc = self.leaky_relu(self.soc_conv(fut_soc_enc))
        # [num_nbrs, C, 15, 1]
        # Add Nonlocal
        fut_soc_enc = self.leaky_relu(self.conv_3x1(fut_soc_enc))
        # [num_nbrs, C, 13, 1]
        fut_soc_enc = self.soc_maxpool(fut_soc_enc)
        fut_soc_enc = fut_soc_enc.view(-1,self.soc_embedding_size)

        goal = self.goal_emb(goal)
        cond = torch.cat([fut_soc_enc,goal],dim=1)
        cond = cond.repeat(self.out_length,1,1)

        #latent dec
        enc_ = torch.randn([fut.shape[1]*25,self.latent_size]).cuda()
        enc_ = enc_.view(self.out_length,-1,enc_.shape[1])
        enc_ = torch.cat([enc_,cond],dim=2)
        enc_ = self.leaky_relu(self.latent_dec(enc_))

        h_dec,_ = self.fut_dec_lstm(enc_,[ego_hist_h,ego_hist_c])
        fut_pred = self.op(h_dec)
        fut_pred = outputActivation(fut_pred)

        return fut_pred, hist_recon

    def condition_forward_(self, c):
        c = c.view(-1,c.shape[2])
        c_mean = self.conditional_mean(c)
        c_std = self.conditional_logvar(c).div(2).exp()
        return c_mean, c_std

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        eps = eps.cuda()
        z = mu+std*eps
        z = z.view(self.out_length,-1,z.shape[1])
        return z





