import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn
from pdb import set_trace as breakpoint
import sys

# from vat import VirtualAdversarialPerturbationGenerator,disable_tracking_bn_stats

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0):
    # feat = fea
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent




def kl_div_consistency(F1, feat, feat_aug):
    bs = feat.size(0)
    all_feat = torch.cat((feat, feat_aug))
    all_logit = F1(all_feat)
    feat_logit, aug_logit = all_logit[:bs], all_logit[bs:]
    kl_div_loss = nn.functional.kl_div(F.log_softmax(aug_logit, dim=1), F.softmax(feat_logit, dim=1))*bs

    return kl_div_loss




def joint_ent(F1, x_out, x_tf_out, lamb=1.0, gamma=1.0,eta=1.0,EPS=sys.float_info.epsilon):
  # has had softmax applied
    out_t1 = F1(x_out, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)

    out_t2 = F1(x_tf_out, reverse=True, eta=-eta)
    out_t2 = F.softmax(out_t2)
    #breakpoint()
    _, k = out_t1.size()
    p_i_j = compute_joint(out_t1, out_t2)
    assert (p_i_j.size() == (k, k))


    loss = - p_i_j * torch.log(p_i_j) 
    #breakpoint()
    loss = loss.sum()

    return lamb*loss

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

# def kl(p, q):
#     return torch.sum(p * F.log((p + 1e-8) / (q + 1e-8))) / float(len(p.data))

def contractive_loss(inputs,temp=0.1):

    similarity = inputs / inputs.norm(dim=1)[:, None]
    #b_norm = b / b.norm(dim=1)[:, None]
    similarity = torch.mm(similarity, similarity.transpose(0,1))/temp  # (2n, 2n)
    similarity[range(len(similarity)), range(len(similarity))] = torch.zeros(len(inputs)).cuda()
    #similarity = F.cosine_similarity(inputs, inputs)/temp  # (2n, 2n)
    N=len(inputs)/2
    for i in range(0,N):
        if i==0:
            loss = -torch.log(similarity[i,i+N]/ torch.sum( similarity[i] )) -torch.log(similarity[i+N,i]/ torch.sum( similarity[i+N] ))
        else:
            loss += -torch.log(similarity[i,i+N]/ torch.sum( similarity[i] ))-torch.log(similarity[i+N,i]/ torch.sum( similarity[i+N] ))
    return loss/(2*N)



class LDSLoss(object):
    def __init__(self, feature_extractor,classifier, xi=1e-6, eps=15, ip=1):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.vap_generator = VirtualAdversarialPerturbationGenerator(self.feature_extractor,self.classifier, xi=xi, eps=eps, ip=ip)
        self.kl_div_loss = nn.KLDivLoss()

    def __call__(self, inputs):
        r_adv, logits = self.vap_generator(inputs)

        adv_inputs = inputs + r_adv
        with disable_tracking_bn_stats(self.feature_extractor):
            with disable_tracking_bn_stats(self.classifier):
                adv_logits = self.feature_extractor(adv_inputs)
                adv_logits = self.classifier(adv_logits)
        lds_loss = self.kl_div_loss(F.log_softmax(adv_logits, dim=1), F.softmax(logits, dim=1))

        return lds_loss



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def regLoss(feat_s1, feat_s2, feat_t, gen_feat, margin=0.1):

    dist_ss = torch.sum((feat_s1 - feat_s2)**2, dim=1)
    dist_gen_t = torch.sum((feat_t - gen_feat)**2, dim=1)
    dist_gen_s1 = torch.sum((feat_s1 - gen_feat)**2, dim=1)
    dist_gen_s2 = torch.sum((feat_s2 - gen_feat)**2, dim=1)
    min_dist_gen_s = torch.min(dist_gen_s1, dist_gen_s2)

    loss1 = F.relu(dist_ss - dist_gen_t + margin).mean()
    loss2 = F.relu(dist_gen_t - min_dist_gen_s + margin).mean()

    # loss1 = F.relu(dist_ss - dist_gen_t + margin)
    # loss2 = F.relu(dist_gen_t - min_dist_gen_s + margin)

    return loss1 + loss2


def regLoss_dummy(feat_s1, feat_s2, feat_t, gen_feat, margin=0.1):

    faet = torch.cat((feat_s1, feat_s2, feat_t))

    # dist_ss = torch.sum((feat_s1 - feat_s2)**2, dim=1)
    # dist_gen_t = torch.sum((feat_t - gen_feat)**2, dim=1)
    # dist_gen_s1 = torch.sum((feat_s1 - gen_feat)**2, dim=1)
    # dist_gen_s2 = torch.sum((feat_s2 - gen_feat)**2, dim=1)
    # min_dist_gen_s = torch.min(dist_gen_s1, dist_gen_s2)

    # loss1 = F.relu(dist_ss - dist_gen_t + margin).mean()
    # loss2 = F.relu(dist_gen_t - min_dist_gen_s + margin).mean()

    return loss1 + loss2




    """
    dSNE Loss
    """
    def __init__(self, margin=None):
        self.margin = margin
        # if margin is not None:
        #     self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        # else:
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, fts, ys, ftt, yt, normalize_feature=False):
        
        if normalize_feature:
            fts = normalize(fts)
            ftt = normalize(ftt)

        ns, nt = fts.size(0), ftt.size(0)
        #breakpoint()
        fts_rpt = fts.unsqueeze(0).expand(nt, ns, -1)
        ftt_rpt = ftt.unsqueeze(1).expand(nt, ns, -1)
        
        # fts_rpt = F.broadcast_to(fts.expand_dims(axis=0), shape=(self._bs_tgt, self._bs_src, self._embed_size))
        # ftt_rpt = F.broadcast_to(ftt.expand_dims(axis=1), shape=(self._bs_tgt, self._bs_src, self._embed_size))

        dist_mat = torch.sum((ftt_rpt - fts_rpt)**2, dim=2)
            
        # breakpoint()        
        # is_pos = yt.expand(N, N).eq(ys.expand(N, N).t())
        # is_neg = yt.expand(N, N).ne(ys.expand(N, N).t())

        ys_rep = ys.unsqueeze(0).expand(nt, ns)
        yt_rep = yt.unsqueeze(1).expand(nt, ns)
        is_pos = yt_rep.eq(ys_rep)
        is_neg = 1-is_pos

        #breakpoint()

        intra_cls_dists = dist_mat * is_pos.float()
        inter_cls_dists = dist_mat * is_neg.float()


        max_dists, _ = torch.max(dist_mat, dim=1, keepdim=True)
        max_dists = max_dists.expand(nt, ns)

        revised_inter_cls_dists = torch.where(is_pos, max_dists, inter_cls_dists)

        max_intra_cls_dist, _ = torch.max(intra_cls_dists, dim=1, keepdim=True)
        min_inter_cls_dist, _ = torch.min(revised_inter_cls_dists, dim=1, keepdim=True)

        # dist_ap, relative_p_inds = torch.max(
        #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # dist_an, relative_n_inds = torch.min(
        #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        
        dist_ap = max_intra_cls_dist.squeeze(1)
        dist_an = min_inter_cls_dist.squeeze(1)


        # breakpoint()        

        # y = dist_an.new().resize_as_(dist_an).fill_(1)
        # if self.margin is not None:
        #     loss = self.ranking_loss(dist_an, dist_ap, y)
        # else:        
        # loss = self.ranking_loss(dist_an - dist_ap, y)
        # return loss

        # breakpoint()

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss




class PrototypeLoss(object):
    def __init__(self, ways=10, trg_shots=3, src_shots=10):
        self.ways = ways
        self.trg_shots = trg_shots
        self.src_shots = src_shots

        label = torch.arange(self.ways).unsqueeze(0).repeat(self.src_shots, 1).t().contiguous().view(-1).squeeze(0)
        # label = torch.arange(self.src_shots).repeat(self.src_shots)

        label = label.type(torch.cuda.LongTensor)
        self.label = label


    def __call__(self, src_feat, target_feat, normalize_feature=False):

        if normalize:
            src_feat = normalize(src_feat, axis=-1)
            proto = normalize(target_feat, axis=-1)

        # breakpoint()

        if self.trg_shots > 1:
            proto = proto.reshape(self.trg_shots, self.ways, -1).mean(dim=0)


        n = src_feat.shape[0]
        m = proto.shape[0]
        src_feat = src_feat.unsqueeze(1).expand(n, m, -1)
        proto = proto.unsqueeze(0).expand(n, m, -1)
        logits = -((src_feat - proto)**2).sum(dim=2)

        # breakpoint()

        loss = F.cross_entropy(logits, self.label)

        return loss




class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        # self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        # if self.use_gpu: targets = targets.cuda()
        targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CrossEntropyKLD(object):
    def __init__(self, num_class=126, mr_weight_kld=0.1):
        self.num_class = num_class 
        self.mr_weight_kld = mr_weight_kld

    def __call__(self, pred, label, mask):
        # valid_reg_num = len(label)
        logsoftmax = F.log_softmax(pred, dim=1)

        kld = torch.sum(-logsoftmax/self.num_class, dim=1)
        ce = (F.cross_entropy(pred, label, reduction='none')*mask).mean()
        kld = (self.mr_weight_kld*kld*mask).mean()

        ce_kld = ce + kld

        return ce_kld




class ProtoLoss(object):    
    def __init__(self, num_classes, feat_dim):
        self.kl_div_loss = nn.KLDivLoss()        
        self.prototypes = torch.FloatTensor(num_classes, feat_dim).cuda()
        self.prototypes.normal_(0, 0.02)
        self.num_classes = num_classes

    def __call__(self, feat, cls_logit, target_feat, target_label, normalize_feature=False):
        if normalize_feature:
            feat = normalize(feat, axis=-1)
            target_feat = normalize(target_feat, axis=-1)


        feat = feat.data
        target_feat = target_feat.data

        proto_logit = torch.mm(feat, self.prototypes.transpose(0,1))
        kl_div_loss = nn.functional.kl_div(F.log_softmax(cls_logit, dim=1), F.softmax(proto_logit, dim=1))*feat.size(0)
        # kl_div_loss = nn.functional.kl_div(F.log_softmax(cls_logit, dim=1), F.softmax(proto_logit, dim=1))

        # breakpoint()

        uni_target_label = torch.unique(target_label, sorted=False)

        # one_hot_support_label = torch.zeros(target_label.size(0), self.num_classes).cuda()
        # one_hot_support_label.scatter_(1, target_label.unsqueeze(1), 1)     
        # breakpoint()        
        # support_feature = self.FeatExemplarAvgBlock(target_feat, one_hot_support_label)  

        # breakpoint()
        target_feat_list = [target_feat[i*3:i*3+3].mean(dim=0, keepdim=True) for i in range(len(uni_target_label))]
        
        # breakpoint()

        target_feat = torch.cat(target_feat_list, dim=0) 

        # breakpoint()

        # self.prototypes[uni_target_label, :] = 
        # self.prototypes[uni_target_label, :] = 0.01*self.prototypes[uni_target_label, :].clone() + 0.99*target_feat.data

        self.prototypes[uni_target_label, :] = target_feat


        return kl_div_loss


    # def FeatExemplarAvgBlock(self, features_train, labels_train):
    #     labels_train_transposed = labels_train.transpose(0,1)

    #     weight_novel = torch.mm(labels_train_transposed, features_train)

    #     breakpoint()

    #     weight_novel = weight_novel.div(labels_train_transposed.sum(dim=1, keepdim=True).expand_as(weight_novel))
    #     return weight_novel
