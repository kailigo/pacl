# pytorch implementation for Transferrable Prototypical Networks for Unsupervised Domain Adaptation
# Sample-level discrepancy loss in Section 3.4 Task-specific Domain Adaptation
# https://arxiv.org/pdf/1904.11227.pdf

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as breakpoint
# from lib.config import cfg



class TpnTaskLoss(nn.Module):
    def __init__(self):
        super(TpnTaskLoss, self).__init__()

    def forward(self, src_feat, trg_feat, src_label, trg_label, trg_feat_un):
        labels = list(src_label.data.cpu().numpy())
        labels = list(set(labels))

        dim = src_feat.size(1)
        center_num = len(labels)
        u_s = torch.zeros(center_num, dim).cuda()
        u_t = torch.zeros(center_num, dim).cuda()
        u_st = torch.zeros(center_num, dim).cuda()

        for i, l in enumerate(labels):
            s_feat = src_feat[src_label == l]
            t_feat = trg_feat[trg_label == l]

            u_s[i, :] = s_feat.mean(dim=0)
            u_t[i, :] = t_feat.mean(dim=0)    
            u_st[i, :] = (s_feat.sum(dim=0) + t_feat.sum(dim=0)) / (s_feat.size(0) + t_feat.size(0))
        
        feats = torch.cat((src_feat, trg_feat), dim=0)
        P_s = torch.matmul(feats, u_s.t())
        P_t = torch.matmul(feats, u_t.t())
        P_st = torch.matmul(feats, u_st.t())

        loss_st = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_t, dim=-1), reduction='mean') + \
            F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_s, dim=-1), reduction='mean')) / 2

        loss_sst = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_st, dim=-1), reduction='mean') + \
            F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_s, dim=-1), reduction='mean')) / 2

        loss_tst = (F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_st, dim=-1), reduction='mean') + \
            F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_t, dim=-1), reduction='mean')) / 2

        tpn_task = (loss_st + loss_sst + loss_tst) / 3
        return tpn_task, ('04. tpn_task loss: ', tpn_task.data.cpu().numpy())





class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()
        self.kernel_mul = 2.0
        # cfg.LOSSES.KERNEL_MUL
        self.kernel_num = 5
        # cfg.LOSSES.KERNEL_NUM
        self.fix_sigma = None

    def guassian_kernel(self, x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        n_samples = x_size + y_size
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        L2_distance = ((tiled_x-tiled_y)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) + 1e-14 for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def compute_mmd(self, x, y):
        x_kernel = self.guassian_kernel(x, x, self.kernel_mul, self.kernel_num, self.fix_sigma)
        y_kernel = self.guassian_kernel(y, y, self.kernel_mul, self.kernel_num, self.fix_sigma)
        xy_kernel = self.guassian_kernel(x, y, self.kernel_mul, self.kernel_num, self.fix_sigma)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def forward(self, x, y, x_labels, y_labels):
        if y_labels is None:            
            mmd = self.compute_mmd(x, y)
            return mmd, ('03. mmd loss: ', mmd.data.cpu().numpy())
        else:
            labels = list(x_labels.data.cpu().numpy())
            labels = list(set(labels))
            pos_num = len(labels)
            neg_num = len(labels) * (len(labels) - 1)
            
            # breakpoint()

            x_c = []
            y_c = []
            n_labels = len(labels)
            for label in labels:
                x_c.append(x[x_labels == label.item()])
                y_c.append(y[y_labels == label.item()])

            xk_c = torch.zeros(n_labels).cuda()
            yk_c = torch.zeros(n_labels).cuda()
            xyk_c = torch.zeros(n_labels, n_labels).cuda()

            # breakpoint()

            for i in range(n_labels):
                
                # breakpoint()
                                
                x_kernel = self.guassian_kernel(x_c[i], x_c[i], self.kernel_mul, self.kernel_num, self.fix_sigma)
                xk_c[i] = x_kernel.mean()

                y_kernel = self.guassian_kernel(y_c[i], y_c[i], self.kernel_mul, self.kernel_num, self.fix_sigma)
                yk_c[i] = y_kernel.mean()

                for j in range(n_labels):
                    xy_kernel = self.guassian_kernel(x_c[i], y_c[j], self.kernel_mul, self.kernel_num, self.fix_sigma)
                    xyk_c[i, j] = xy_kernel.mean()

            xk_c_sum = xk_c.sum()
            yk_c_sum = yk_c.sum()
            xyk_c_diag = torch.eye(n_labels, n_labels).cuda() * xyk_c
            xyk_c_antidiag = (1 - torch.eye(n_labels, n_labels).cuda()) * xyk_c

            mmd = (xk_c_sum + yk_c_sum - 2 * xyk_c_diag.sum()) / pos_num

            # mmd -= cfg.LOSSES.MMD_NEG_WEIGHT * ( (n_labels - 1) * (xk_c_sum + yk_c_sum) - 2 * xyk_c_antidiag.sum() )  / neg_num
            # mmd -= cfg.LOSSES.MMD_NEG_WEIGHT * ( (n_labels - 1) * (xk_c_sum + yk_c_sum) - 2 * xyk_c_antidiag.sum() )  / neg_num
            
            return mmd
            # mmd, ('03. mmd loss: ', mmd.data.cpu().numpy())



class TpnTaskLoss_Non_Param(nn.Module):
    def __init__(self):
        super(TpnTaskLoss_Non_Param, self).__init__()
        self.MMDLoss = MMDLoss()


    def forward(self, src_feat, trg_feat, src_label, trg_label, trg_feat_un):
        # mmd_loss = self.MMDLoss(src_feat, trg_feat, src_label, trg_label)
        # labels = list(src_label.data.cpu().numpy())
        # labels = list(set(labels))

        labels = torch.unique(src_label)

        ns, dim = src_feat.size(0), src_feat.size(1)
        nt = trg_feat.size(0)

        center_num = len(labels)
        u_s = torch.zeros(center_num, dim).cuda()
        u_t = torch.zeros(center_num, dim).cuda()
        u_st = torch.zeros(center_num, dim).cuda()

        for i, l in enumerate(labels):
            s_feat = src_feat[src_label == l]
            t_feat = trg_feat[trg_label == l]

            u_s[i, :] = s_feat.mean(dim=0)
            u_t[i, :] = t_feat.mean(dim=0)    
            u_st[i, :] = (s_feat.sum(dim=0) + t_feat.sum(dim=0)) / (s_feat.size(0) + t_feat.size(0))
        

        mmd_loss = (F.mse_loss(u_s, u_t) + F.mse_loss(u_s, u_st) + F.mse_loss(u_t, u_st)) / 3


        feats = torch.cat((src_feat, trg_feat, trg_feat_un), dim=0)
        P_s = torch.matmul(feats, u_s.t())
        P_t = torch.matmul(feats, u_t.t())
        P_st = torch.matmul(feats, u_st.t())

        # breakpoint()

        src_uni_label, src_uni_indx = torch.unique(src_label, sorted=True, return_inverse=True)
        trg_uni_label, trg_uni_indx = torch.unique(trg_label, sorted=True, return_inverse=True)


        loss_supv_s = F.cross_entropy(P_s[:ns], src_uni_indx)
        loss_supv_t = F.cross_entropy(P_t[ns:ns+nt], trg_uni_indx)

        # breakpoint()

        loss_st = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_t, dim=-1), reduction='elementwise_mean') + \
            F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_s, dim=-1), reduction='elementwise_mean')) / 2

        loss_sst = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_st, dim=-1), reduction='elementwise_mean') + \
            F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_s, dim=-1), reduction='elementwise_mean')) / 2

        loss_tst = (F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_st, dim=-1), reduction='elementwise_mean') + \
            F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_t, dim=-1), reduction='elementwise_mean')) / 2


        loss_tpn = (loss_st + loss_sst + loss_tst) / 3

        # breakpoint()

        return loss_tpn, loss_supv_s, loss_supv_t, mmd_loss

        # ('04. tpn_task loss: ', tpn_task.data.cpu().numpy())


    def forward_test(self, trg_feat):

        # proto_t_norm = F.normalize(self.proto_t, p=2, dim=self.proto_t.dim()-1, eps=1e-12)
        # un_trg_feats_norm = F.normalize(trg_feat, p=2, dim=trg_feat.dim()-1, eps=1e-12)
        # trg_score = torch.mm(un_trg_feats_norm, proto_t_norm)

        # trg_score = F.cosine_similarity(trg_feat, self.proto_t, dim=-1)
        return trg_score



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x




class TpnTaskLoss_Simple(object):
    # def __init__(self):
    #     super(TpnTaskLoss_Simple, self).__init__()
        # self.MMDLoss = MMDLoss()
        
    def __call__(self, src_feat, trg_feat, src_label, trg_label, trg_feat_un, do_normalization=False):

        if do_normalization:
            # src_feat = F.normalize(src_feat)
            # trg_feat = F.normalize(trg_feat)
            # trg_feat_un = F.normalize(trg_feat_un)

            src_feat = normalize(src_feat)
            trg_feat = normalize(trg_feat)
            trg_feat_un = normalize(trg_feat_un)


        labels = torch.unique(src_label)

        ns, dim = src_feat.size(0), src_feat.size(1)
        nt = trg_feat.size(0)

        center_num = len(labels)
        u_s = torch.zeros(center_num, dim).cuda()
        u_t = torch.zeros(center_num, dim).cuda()


        # breakpoint()

        for i, l in enumerate(labels):
            s_feat = src_feat[src_label == l]
            t_feat = trg_feat[trg_label == l]

            u_s[i, :] = s_feat.mean(dim=0)
            u_t[i, :] = t_feat.mean(dim=0)    


        # feats = torch.cat((src_feat, trg_feat, trg_feat_un), dim=0)
        feats = trg_feat_un
        P_s = torch.matmul(feats, u_s.t())
        P_t = torch.matmul(feats, u_t.t())

        # src_uni_label, src_uni_indx = torch.unique(src_label, sorted=True, return_inverse=True)
        # trg_uni_label, trg_uni_indx = torch.unique(trg_label, sorted=True, return_inverse=True)

        # loss_supv_s = F.cross_entropy(P_s[:ns], src_uni_indx)
        # loss_supv_t = F.cross_entropy(P_t[ns:ns+nt], trg_uni_indx)

        # breakpoint()


        loss_st = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_t, dim=-1), reduction='elementwise_mean') 
             +  F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_s, dim=-1), reduction='elementwise_mean')) * center_num / 2

        # loss_sst = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_st, dim=-1), reduction='elementwise_mean') + \
        #     F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_s, dim=-1), reduction='elementwise_mean')) / 2
        # loss_tst = (F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_st, dim=-1), reduction='elementwise_mean') + \
        #     F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_t, dim=-1), reduction='elementwise_mean')) / 2

        # loss_tpn = (loss_st + loss_sst + loss_tst) / 3
        loss_tpn = loss_st

        return loss_tpn

        # return loss_tpn, loss_supv_s, loss_supv_t, mmd_loss


    # def forward_test(self, trg_feat):

    #     return trg_score






class TpnTaskLoss_Non_Param_Simple(nn.Module):
    def __init__(self):
        super(TpnTaskLoss_Non_Param_Simple, self).__init__()


    def forward(self, src_feat, trg_feat, src_label, trg_label):

        labels = torch.unique(src_label)

        ns, dim = src_feat.size(0), src_feat.size(1)
        nt = trg_feat.size(0)

        center_num = len(labels)
        u_t = torch.zeros(center_num, dim).cuda()

        for i, l in enumerate(labels):
            t_feat = trg_feat[trg_label == l]
            u_t[i, :] = t_feat.mean(dim=0)    
        

        feats = torch.cat((src_feat, trg_feat), dim=0)       
        label_combo = torch.cat((src_label, trg_label), dim=0)       
        # label_combo = src_label

        # breakpoint()

        P_t = torch.matmul(feats, u_t.t())
        
        src_uni_indx = label_combo
        # src_uni_label, src_uni_indx = torch.unique(label_combo, sorted=True, return_inverse=True)

        loss_supv_s = F.cross_entropy(P_t, src_uni_indx)
        
        return loss_supv_s


    def forward_test(self, trg_feat):

        return trg_score




class TpnTaskLoss_Param(nn.Module):
    def __init__(self, num_class=64, dim=4096):
        super(TpnTaskLoss_Param, self).__init__()                    
        self.proto_s = nn.Parameter(torch.FloatTensor(dim, num_class), requires_grad=True)
        self.proto_t = nn.Parameter(torch.FloatTensor(dim, num_class), requires_grad=True)
        self.proto_s.data.normal_(0, 0.02)
        self.proto_t.data.normal_(0, 0.02)

        
    def forward_standard(self, src_feat, trg_feat, trg_feat_un):
        nf, dim = src_feat.size()
        
        un_src_feats = torch.cat((src_feat, trg_feat_un), dim=0)
        un_trg_feats = torch.cat((trg_feat, trg_feat_un), dim=0)

        # breakpoint()

        proto_s_norm = F.normalize(self.proto_s, p=2, dim=self.proto_s.dim()-1, eps=1e-12)
        proto_t_norm = F.normalize(self.proto_t, p=2, dim=self.proto_t.dim()-1, eps=1e-12)

        un_src_feats_norm = F.normalize(un_src_feats, p=2, dim=un_src_feats.dim()-1, eps=1e-12)
        un_trg_feats_norm = F.normalize(un_trg_feats, p=2, dim=un_trg_feats.dim()-1, eps=1e-12)

        src_score = torch.mm(un_src_feats_norm, proto_s_norm)
        trg_score = torch.mm(un_trg_feats_norm, proto_t_norm)

        # src_score = F.cosine_similarity(un_src_feats.unsqueeze(0), self.proto_s, dim=1)
        # trg_score = F.cosine_similarity(un_trg_feats, self.proto_t, dim=0)

        cls_src_score, un_src_score = src_score[:nf], src_score[nf:] 
        cls_trg_score, un_trg_score = trg_score[:nf], trg_score[nf:] 

        # breakpoint()

        loss_st = (F.kl_div(F.log_softmax(un_src_score, dim=-1), F.softmax(un_trg_score, dim=-1), size_average=False) + \
            F.kl_div(F.log_softmax(un_trg_score, dim=-1), F.softmax(un_src_score, dim=-1), size_average=False)) / 2

        tpn_loss = loss_st / trg_feat_un.size(0)

        return cls_src_score, cls_trg_score, tpn_loss



    def forward(self, src_feat, trg_feat, trg_feat_un):
        nf, dim = src_feat.size()
        
        feat = torch.cat((src_feat, trg_feat, trg_feat_un), dim=0)

        # feat_norm = F.normalize(self.proto_s, p=2, dim=self.proto_s.dim()-1, eps=1e-12)
        # un_src_feats = torch.cat((src_feat, trg_feat_un), dim=0)
        # un_trg_feats = torch.cat((trg_feat, trg_feat_un), dim=0)

        proto_s_norm = F.normalize(self.proto_s, p=2, dim=self.proto_s.dim()-1, eps=1e-12)
        proto_t_norm = F.normalize(self.proto_t, p=2, dim=self.proto_t.dim()-1, eps=1e-12)

        feat_norm = F.normalize(feat, p=2, dim=feat.dim()-1, eps=1e-12)
        # un_trg_feats_norm = F.normalize(un_trg_feats, p=2, dim=un_trg_feats.dim()-1, eps=1e-12)

        src_score = torch.mm(un_src_feats_norm, proto_s_norm)
        trg_score = torch.mm(un_trg_feats_norm, proto_t_norm)

        
        src_score_src, trg_score_src, un_score_src = src_score[:nf], src_score[nf:nf+nf], src_score[nf+nf:]
        src_score_trg, trg_score_trg, un_score_trg = trg_score[:nf], trg_score[nf:nf+nf], trg_score[nf+nf:]
        
        
        loss_st = (F.kl_div(F.log_softmax(un_src_score, dim=-1), F.softmax(un_trg_score, dim=-1), size_average=False) + \
            F.kl_div(F.log_softmax(un_trg_score, dim=-1), F.softmax(un_src_score, dim=-1), size_average=False)) / 2

        tpn_loss = loss_st / trg_feat_un.size(0)

        return cls_src_score, cls_trg_score, tpn_loss



    def forward_test(self, trg_feat):

        proto_t_norm = F.normalize(self.proto_t, p=2, dim=self.proto_t.dim()-1, eps=1e-12)
        un_trg_feats_norm = F.normalize(trg_feat, p=2, dim=trg_feat.dim()-1, eps=1e-12)
        trg_score = torch.mm(un_trg_feats_norm, proto_t_norm)

        # trg_score = F.cosine_similarity(trg_feat, self.proto_t, dim=-1)
        return trg_score

