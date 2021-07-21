import torch.nn as nn
import torch.nn.functional as fn
from torch_scatter import scatter
import torch_scatter

from .pytorch_utils import MixedDropout, MixedLinear

class RoutingLayer(nn.Module):
    """
    This class describes a interest routing layer.
    """
    def __init__(self, dim, num_caps, init_mtd, attention, beta, is_xNorm):#-----0917
        """
        :dim: the output dimension of this layer
        :num_caps: number of interest capsules
        """
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self.init_mtd, self.attention, self.beta = init_mtd, attention, beta
        self.is_xNorm = is_xNorm ####~~~~~

    def forward(self, x_nb, ppr, row_idx, col_idx, x_idx, max_iter):#tar_nod_feat, tar_nb_feat, tar_ppr, max_iter
        """
        forward propagation

        :param      x:          feature vectors of target nodes
        :type       x:          torch tensor, size: (n,d)
        :param      z:          The feature of target nodes' neighbors
        :type       z:          torch tensor, len: (n*m, d)
        :param      nb_ppr:     The neighbors' ppr with respect to target nodes
        :type       nb_ppr:     Torch tensor, size: (n,m)
        :param      max_iter:   The maximum routing iterator
        :type       max_iter:   int

        :returns:   final representation vectors of target nodes
        :rtype:     torch tensor
        """

        d, k, delta_d = self.d, self.k, self.d // self.k
        
        if self.is_xNorm:
            x_nb = fn.normalize(x_nb.view(-1, k, delta_d), dim=2).view(-1, d) #-----0917
        
        if self.init_mtd == 0:
            u = x_nb[x_idx]
        elif self.init_mtd==1:#self.init_mtd == 1# ---initialize u as mean of neighbors' feature ---
            u = torch_scatter.scatter(x_nb[col_idx] , row_idx, dim=0, reduce='sum')
        else: # self.init_mtd == 2# ---initialize u as sum of neighbors' feature weighted by ppr #-----0916
            u = torch_scatter.scatter(x_nb[col_idx]* ppr.view(-1,1) , row_idx, dim=0, reduce='sum') #-----0916
            
        for clus_iter in range(max_iter):
            p = ((u[row_idx] * x_nb[col_idx]).view(-1,k,delta_d)).sum(2)
            if self.attention == 0: # attention = softmax(feature_sim * ppr)
                p = p * ppr.view(-1,1)
            elif self.attention == 1: # self.attention == 1 -- attention = softmax( beta * feature_sim + (1-beta) * softmax(ppr) )#-----0916
                for i in range(k):
                    p[:,i] = torch_scatter.composite.scatter_softmax(p[:,i], row_idx)
                p = self.beta * p + (1 - self.beta) * ppr.view(-1,1)
#             else: # self.attention == 2 -- attention = feature_sim #-----0916
                
            for i in range(k):
                p[:,i] = torch_scatter.composite.scatter_softmax(p[:,i], row_idx)
            u = torch_scatter.scatter(x_nb[col_idx].view(-1,k,delta_d)*(p.view(-1,k,1).repeat(1,1,delta_d)), row_idx, dim=0, reduce='sum')
            if clus_iter < max_iter - 1 and self.is_xNorm:#-----0917
                u = fn.normalize(u, dim=2) 
            u = u.view(-1, d)
        return u.view(-1, d)


class COSAL(nn.Module):
    def __init__(self, nfeat, nclass, hidden_size, ncaps, init_mtd, attention, beta, dropout, rouit, isReg, cpu, is_xNorm):#-----0917
        """
        Constructs a new instance.

        :param      nfeat:    The nfeat
        :type       nfeat:    { type_description }
        :param      nclass:   The nclass
        :type       nclass:   { type_description }
        :param      hyperpm:  The hyperpm
        :type       hyperpm:  { type_description }
        """
        super(COSAL, self).__init__()
        self.k, self.d, self.delta_d = ncaps, hidden_size * ncaps, hidden_size
        pca = MixedLinear(nfeat, self.d, bias=True)
        self.add_module('pca', pca)
        conv = RoutingLayer(self.d, self.k, init_mtd, attention, beta, is_xNorm)
        self.add_module('conv', conv)
        mlp =  nn.Linear(self.d, nclass)
        self.add_module('mlp', mlp)
        self._dropout = MixedDropout(dropout)
        self.rouit = rouit
        self.isReg = isReg


    def forward(self, x_nb, ppr, row_idx, col_idx, x_idx):
        """
        x_nb: 当前batch的attr_matrix
        x_idx: x_nb中哪些是target node 自身的, bool tensor
        row_idx, col_idx: 表征各个target nodes拥有哪些nb nodes
        """
        d = x_nb.size(1)
        x_nb = fn.relu(self.pca(x_nb))
        ppr = torch_scatter.composite.scatter_softmax(ppr,row_idx)# whether is necccessary ?
        x = self._dropout(fn.relu(self.conv(x_nb, ppr, row_idx, col_idx, x_idx, self.rouit)))
        if self.isReg and self.training:
            x_reshaped = x.view(-1, self.k, self.delta_d).contiguous()
        x = self.mlp(x.view(-1, self.d))
        prob = fn.log_softmax(x, dim=1)
        if self.isReg and self.training:
            return prob, x_reshaped
        else:
            return prob
