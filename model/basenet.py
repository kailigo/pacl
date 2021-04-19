from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
# import ipdb


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        #ipdb.set_trace()
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()

        self.fc = nn.Linear(inc, num_class, bias=False)
        # self.fc = nn.Linear(inc, num_class)

        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        # x_out = self.fc(x)
        return x_out


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x



class Predictor_Proto(object):    
    def __init__(self, num_class, inc, temp=0.05):
        super(Predictor_Proto, self).__init__()

        self.prototypes = torch.FloatTensor(num_class, inc).cuda()
        self.prototypes.normal_(0, 0.02)
        # self.num_classes = num_class
        self.temp = temp

    def __call__(self, feat, reverse=False, eta=0.1, update_proto=False, target_feat=None, target_label=None):

        # reverse
        if reverse:
            feat = grad_reverse(feat, eta)

        feat = normalize(feat, axis=-1)

        # if normalize_feature:
                    
        feat = feat.data        
        proto_logit = torch.mm(feat, self.prototypes.transpose(0,1)) / self.temp

        # kl_div_loss = nn.functional.kl_div(F.log_softmax(cls_logit, dim=1), F.softmax(proto_logit, dim=1))*feat.size(0)
        # kl_div_loss = nn.functional.kl_div(F.log_softmax(cls_logit, dim=1), F.softmax(proto_logit, dim=1))
        # breakpoint()

        if update_proto:

            target_feat = target_feat.data

            target_feat = normalize(target_feat, axis=-1)
            uni_target_label = torch.unique(target_label, sorted=False)
            target_feat_list = [target_feat[i*3:i*3+3].mean(dim=0, keepdim=True) for i in range(len(uni_target_label))]
            target_feat = torch.cat(target_feat_list, dim=0) 
            self.prototypes[uni_target_label, :] = target_feat

        return proto_logit


        # one_hot_support_label = torch.zeros(target_label.size(0), self.num_classes).cuda()
        # one_hot_support_label.scatter_(1, target_label.unsqueeze(1), 1)     
        # breakpoint()        
        # support_feature = self.FeatExemplarAvgBlock(target_feat, one_hot_support_label)  

        # breakpoint()        
        
        # breakpoint()


        # breakpoint()

        # self.prototypes[uni_target_label, :] = 
        # self.prototypes[uni_target_label, :] = 0.01*self.prototypes[uni_target_label, :].clone() + 0.99*target_feat.data

        


        


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class fc_head(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(fc_head, self).__init__()
        self.fc = nn.Linear(inc, num_class)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x):
        #x = F.normalize(x)
        x_out = self.fc(x) 
        return x_out

class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 2)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out        


class Distance_metric(nn.Module):
    def __init__(self, num_class=2, inc=4096, temp=0.05):
        super(Distance_metric, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x_out = self.fc(x) 
        return x_out