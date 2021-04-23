import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist, Imagelists_VISDA_Src_Trg_Wt_Unl, Imagelists_VISDA_Twice,Imagelists_VISDA_un
from loaders.data_list import DomainArrayDataset, DomainArrayDataset_Triplet, Imagelists_VISDA_Target_Labeled,Imagelists_VISDA_Target_Labeled_self

from loaders.data_list import Imagelists_VISDA_from_list, Imagelists_VISDA_Target_Labeled_from_list, Imagelists_VISDA_un_from_list, Imagelists_VISDA_un_3u

import pickle
from pdb import set_trace as breakpoint
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random

from PIL import Image
import copy
import pdb
import torchnet as tnt
from torch.utils.data.dataloader import default_collate

from loaders.data_list import make_dataset_fromlist

from .randaugment import RandAugmentMC



def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter,rnd_gray])
    return color_distort


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))



def return_dataset_hard(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset


    if args.dataset == 'visda':
        image_set_file_s = \
            os.path.join(base_path,
                         'image_list.txt')
        
        image_set_file_t = \
            os.path.join(base_path,
                         'set1_labeled_target_images_sketch_3.txt')
        
        image_set_file_t_val = \
            os.path.join(base_path,
                         'set1_unlabeled_target_images_sketch_3.txt')

        image_set_file_unl = \
            os.path.join(base_path,
                        'set1_unlabeled_target_images_sketch_3.txt')

    else:
        image_set_file_s = \
            os.path.join(base_path,
                         'labeled_source_images_' +
                         args.source + '.txt')
        image_set_file_t = \
            os.path.join(base_path,
                         'set2_labeled_target_images_' +
                         args.target + '_%d.txt' % (args.num))
        image_set_file_t_val = \
            os.path.join(base_path,
                         'validation_target_images_' +
                         args.target + '_3.txt')
        image_set_file_unl = \
            os.path.join(base_path,
                         'set2_unlabeled_target_images_' +
                         args.target + '_%d.txt' % (args.num))




    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    if args.dataset == 'visda':

        source_dataset = Imagelists_VISDA(image_set_file_s, root=root+'/train',
                                          transform=data_transforms['train'])

        target_dataset = Imagelists_VISDA(image_set_file_t, root=root+'/validation',
                                          transform=data_transforms['train'])

        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root+'/validation',
                                              transform=data_transforms['test'])

        target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root+'/validation',
                                              transform=data_transforms['train'])

        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root+'/validation',
                                               transform=data_transforms['test'])

    else:
        source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                          transform=data_transforms['train'])

        target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                          transform=data_transforms['train'])

        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                              transform=data_transforms['test'])

        target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                              transform=data_transforms['train'])

        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                               transform=data_transforms['test'])




    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24


    nw = 12
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=nw, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=nw,
                                    shuffle=True, drop_last=True)


    source_loader_eval = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=nw, shuffle=True,
                                                drop_last=True)
    target_loader_eval = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=nw,
                                    shuffle=True, drop_last=True)


    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list






def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset




    if args.dataset == 'visda':
        image_set_file_s = \
            os.path.join(base_path,
                         'image_list.txt')
        
        image_set_file_t = \
            os.path.join(base_path,
                         'set1_labeled_target_images_sketch_3.txt')
        
        image_set_file_t_val = \
            os.path.join(base_path,
                         'set1_unlabeled_target_images_sketch_3.txt')

        image_set_file_unl = \
            os.path.join(base_path,
                        'set1_unlabeled_target_images_sketch_3.txt')

    else:
        image_set_file_s = \
            os.path.join(base_path,
                         'labeled_source_images_' +
                         args.source + '.txt')
        image_set_file_t = \
            os.path.join(base_path,
                         'set2_labeled_target_images_' +
                         args.target + '_%d.txt' % (args.num))
        image_set_file_t_val = \
            os.path.join(base_path,
                         'validation_target_images_' +
                         args.target + '_3.txt')
        image_set_file_unl = \
            os.path.join(base_path,
                         'set2_unlabeled_target_images_' +
                         args.target + '_%d.txt' % (args.num))




    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            # RandAugmentMC(n=2, m=10),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



    if args.dataset == 'visda':

        source_dataset = Imagelists_VISDA(image_set_file_s, root=root+'/train',
                                          transform=data_transforms['train'])

        target_dataset = Imagelists_VISDA(image_set_file_t, root=root+'/validation',
                                          transform=data_transforms['train'])

        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root+'/validation',
                                              transform=data_transforms['test'])

        target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root+'/validation',
                                              transform=data_transforms['train'])

        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root+'/validation',
                                               transform=data_transforms['test'])

    else:
        source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                          transform=data_transforms['train'])

        target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                          transform=data_transforms['train'])

        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                              transform=data_transforms['test'])

        target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                              transform=data_transforms['train'])

        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                               transform=data_transforms['test'])



    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24


    nw = 12
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=nw, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=nw,
                                    shuffle=True, drop_last=True)


    source_loader_eval = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=nw, shuffle=True,
                                                drop_last=True)
    target_loader_eval = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=nw,
                                    shuffle=True, drop_last=True)


    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list



def load_pickle():
    with open('dict_path2img.pickle', 'rb') as config_dictionary_file:
        dict_path2img = pickle.load(config_dictionary_file)
    
    return dict_path2img




class GeneratorSampler_dummy(Sampler):
    def __init__(self, num_of_class, source_label, target_label, num_per_class_src, num_per_class_trg):
        
        self.num_instances = num_per_class_src
        self.num_pids_per_batch = num_of_class
        
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.index_dic_trg = defaultdict(list)
        for index, pid in enumerate(target_label):
            self.index_dic_trg[pid].append(index)

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        


    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        final_pid = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)


    def __len__(self):
        return self.length




class GeneratorSampler(Sampler):
    def __init__(self, num_of_class, source_label, num_per_class_src):
        
        self.num_instances = num_per_class_src
        self.num_pids_per_batch = num_of_class
        
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())


        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        # breakpoint()

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        final_pid = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)


    def __len__(self):
        return self.length




class RandomIdentitySampler_alignedreid(Sampler):
    def __init__(self, num_of_class, source_label, num_per_class_src):        

        self.num_instances = num_per_class_src
        self.num_pids_per_batch = num_of_class
        
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)


    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances



class RandomIdentitySampler_alignedreid_pseudo1(Sampler):
    def __init__(self, num_of_class, source_label, target_label, num_per_class_src, num_per_class_trg,  ways):

        self.index_dic_src = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic_src[pid].append(index)        
        num_of_all_source = len(source_label)

        self.index_dic_trg = defaultdict(list)
        for index, pid in enumerate(target_label):
            self.index_dic_trg[pid].append(index+num_of_all_source)

        self.num_per_class_src = num_per_class_src
        self.num_per_class_trg = num_per_class_trg
        self.num_of_class = num_of_class
        self.classes = list(self.index_dic_src.keys())
        self.num_identities = len(self.classes)
        self.ways = ways


    def __iter__(self):        

        class_list = list(range(self.num_identities))
        random.shuffle(class_list)

        # batch_src = []
        # batch_trg = []
        ret = []
        for j in class_list:
            src_pid = self.index_dic_src[j]
            trg_pid = self.index_dic_trg[j]

            replace1 = False if len(trg_pid) >= self.num_per_class_trg else True
            replace2 = False if len(src_pid) >= self.num_per_class_src else True

            src_t = np.random.choice(src_pid, size=self.num_per_class_src, replace=replace2)
            trg_t = np.random.choice(trg_pid, size=self.num_per_class_trg, replace=replace1)
            
            ret.extend(src_t)
            ret.extend(trg_t)

            # batch_src.extend(src_t)
            # batch_trg.extend(trg_t)

        # batch_src.extend(batch_trg)
        return iter(ret)

    def __len__(self):
        # return self.num_identities * self.num_per_class_trg
        # return 10000000
        return self.num_identities*(self.num_per_class_trg+self.num_per_class_src)




class RandomIdentitySampler_alignedreid1(Sampler):
    def __init__(self, num_of_class, source_label, num_per_class_src):

        self.index_dic_src = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic_src[pid].append(index)        
        # num_of_all_source = len(source_label)

        # self.index_dic_trg = defaultdict(list)
        # for index, pid in enumerate(target_label):
        #     self.index_dic_trg[pid].append(index+num_of_all_source)

        self.num_per_class_src = num_per_class_src
        # self.num_per_class_trg = num_per_class_trg
        self.num_of_class = num_of_class
        self.classes = list(self.index_dic_src.keys())


    def __iter__(self):        
        class_list = np.random.choice(self.classes, self.num_of_class, replace=False)

        batch_src = []
        batch_trg = []
        for j in class_list:
            src_pid = self.index_dic_src[j]
            # trg_pid = self.index_dic_trg[j]

            # replace = False if len(trg_pid) >= self.num_per_class_trg else True

            src_t = np.random.choice(src_pid, size=self.num_per_class_src, replace=False)
            # trg_t = np.random.choice(trg_pid, size=self.num_per_class_trg, replace=replace)
            
            batch_src.extend(src_t)
            # batch_trg.extend(trg_t)

        # batch_src.extend(batch_trg)
        return iter(batch_src)

    def __len__():
        return 1





# class RandomIdentitySampler_alignedreid_pseudo(Sampler):
#     def __init__(self, num_of_class, source_label, target_label, num_per_class_src, num_per_class_trg):        

#         self.num_instances = num_per_class_src
#         self.num_pids_per_batch = num_of_class
        
#         self.index_dic = defaultdict(list)
#         for index, pid in enumerate(source_label):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         self.num_identities = len(self.pids)


#     def __iter__(self):

#         indices = torch.randperm(self.num_identities)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             replace = False if len(t) >= self.num_instances else True
#             t = np.random.choice(t, size=self.num_instances, replace=replace)
#             ret.extend(t)
#         return iter(ret)

#     def __len__(self):
#         return self.num_identities * self.num_instances




class RandomIdentitySampler_alignedreid_pseudo(Sampler):
    def __init__(self, num_of_class, source_label, target_label, num_per_class_src, num_per_class_trg):

        self.index_dic_src = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic_src[pid].append(index)        
        num_of_all_source = len(source_label)

        self.index_dic_trg = defaultdict(list)
        for index, pid in enumerate(target_label):
            self.index_dic_trg[pid].append(index+num_of_all_source)

        self.num_per_class_src = num_per_class_src
        self.num_per_class_trg = num_per_class_trg
        self.num_of_class = num_of_class
        self.classes = list(self.index_dic_src.keys())


    def __iter__(self):        
        class_list = np.random.choice(self.classes, self.num_of_class, replace=False)

        batch_src = []
        batch_trg = []
        for j in class_list:
            src_pid = self.index_dic_src[j]
            trg_pid = self.index_dic_trg[j]

            replace = False if len(trg_pid) >= self.num_per_class_trg else True

            src_t = np.random.choice(src_pid, size=self.num_per_class_src, replace=False)
            trg_t = np.random.choice(trg_pid, size=self.num_per_class_trg, replace=replace)
            
            batch_src.extend(src_t)
            batch_trg.extend(trg_t)


        batch_src.extend(batch_trg)
        return iter(batch_src)

    def __len__():
        return 1


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2




def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')





# class RandomIdentitySampler_alignedreid_src_trg(Sampler):
#     def __init__(self, num_of_class, source_label, target_label, num_per_class_src, num_per_class_trg):

#         self.index_dic_src = defaultdict(list)
#         for index, pid in enumerate(source_label):
#             self.index_dic_src[pid].append(index)        
#         num_of_all_source = len(source_label)

#         self.index_dic_trg = defaultdict(list)
#         for index, pid in enumerate(target_label):
#             self.index_dic_trg[pid].append(index+num_of_all_source)

#         self.num_per_class_src = num_per_class_src
#         self.num_per_class_trg = num_per_class_trg
#         self.num_of_class = num_of_class
#         self.classes = list(self.index_dic_src.keys())
#         self.num_identities = len(self.classes)


#     def __iter__(self):        
#         # class_list = np.random.choice(self.classes, self.num_of_class, replace=False)

#         class_list = list(range(self.num_identities))
#         random.shuffle(class_list)

#         batch_src = []
#         batch_trg = []
#         for j in class_list:
#             src_pid = self.index_dic_src[j]
#             trg_pid = self.index_dic_trg[j]

#             replace1 = False if len(trg_pid) >= self.num_per_class_trg else True
#             replace2 = False if len(src_pid) >= self.num_per_class_src else True

#             # breakpoint()

#             src_t = np.random.choice(src_pid, size=self.num_per_class_src, replace=replace2)
#             trg_t = np.random.choice(trg_pid, size=self.num_per_class_trg, replace=replace1)
            
#             batch_src.extend(src_t)
#             batch_trg.extend(trg_t)

#         batch_src.extend(batch_trg)
#         return iter(batch_src)

#     def __len__():
#         return self.num_identities * self.num_per_class_src




def return_dataset_balance_fast_src_trg(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None
        
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['val'])

    # train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    # target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
    #                                 transform=data_transforms['train'], dict_path2img=dict_path2img) 

    src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
    trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)


    # breakpoint()

    labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
    labels = np.concatenate((src_labels, trg_train_labels))
    labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=root, transform=data_transforms['train'])


    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots
    nw = 12

    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
    #     num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
    #     source_label=train_dataset.labels, num_per_class_src=args.src_shots))

    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.ways*(args.src_shots+args.trg_shots), 
            num_workers=12, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid_pseudo1(num_of_class=args.ways, 
                source_label=src_labels, target_label=trg_train_labels, 
                num_per_class_src=args.src_shots, num_per_class_trg=args.trg_shots, ways=args.ways))


    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
    #                                 num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs*2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return labeled_data_loader, target_loader_val, target_loader_test, target_loader_unl, class_list



def return_dataset_balance_fast_src_trg_hard(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None
        
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['val'])

    # train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    # target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
    #                                 transform=data_transforms['train'], dict_path2img=dict_path2img) 

    src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
    trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)


    # breakpoint()

    labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
    labels = np.concatenate((src_labels, trg_train_labels))
    labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=root, transform=data_transforms['train'])


    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots
    nw = 12

    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
    #     num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
    #     source_label=train_dataset.labels, num_per_class_src=args.src_shots))

    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.ways*(args.src_shots+args.trg_shots), 
            num_workers=12, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid_pseudo1(num_of_class=args.ways, 
                source_label=src_labels, target_label=trg_train_labels, 
                num_per_class_src=args.src_shots, num_per_class_trg=args.trg_shots, ways=args.ways))


    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
    #                                 num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs*2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return labeled_data_loader, target_loader_val, target_loader_test, target_loader_unl, class_list






def return_dataset_balance_fast(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None
        
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['val'])

    train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
                                    transform=data_transforms['train'], dict_path2img=dict_path2img) 

    # self.src_imgs, self.src_labels = make_dataset_fromlist(image_set_file_s)
    # self.trg_train_imgs, self.trg_train_labels = make_dataset_fromlist(image_set_file_t)

    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots
    nw = 12

    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
        num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
        source_label=train_dataset.labels, num_per_class_src=args.src_shots))


    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
    #                                 num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs*2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list




class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor



class DataLoader1(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 drop_last=True,
                 sampler=None,
                 crop_size= None,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            #Denormalize(mean_pix, std_pix),
            #transforms.RandomResizedCrop(crop_size,scale=(0.08, 1.0)),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        #aa=func1(img[0])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
        def _load_function(idx):
            idx = idx % len(self.dataset)
            img0, label = self.dataset[idx]
            #ipdb.set_trace()
            rotated_imgs = [
                self.transform(img0),
                self.transform(img0)
            ]

            return torch.stack(rotated_imgs, dim=0), label.repeat(2)
        def _collate_fun(batch):
            batch = default_collate(batch)
            #pdb.set_trace()
            #assert(len(batch)==2)
            batch_size, rotations, channels, height, width = batch[0].size()
            batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
            batch[1] = batch[1].view([batch_size*rotations])
            return batch

        #_collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        #pdb.set_trace()
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle, drop_last=self.drop_last,sampler=self.sampler)
        return data_loader


    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size




class DATA():
    def __init__(self, args):

        base_path = './data/txt/%s' % args.dataset
        root = './data/%s/' % args.dataset
        image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
        image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
        image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
        image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

        
        self.src_imgs, self.src_labels = make_dataset_fromlist(image_set_file_s)
        self.trg_train_imgs, self.trg_train_labels = make_dataset_fromlist(image_set_file_t)
        self.trg_val_imgs, self.trg_val_labels = make_dataset_fromlist(image_set_file_t_val)    
        self.trg_test_imgs, self.trg_test_labels = make_dataset_fromlist(image_set_file_unl)


        if args.net == 'alexnet':
            crop_size = 227
        else:
            crop_size = 224

        self.data_transforms = self.get_transforms(crop_size)
        
        train_dataset = Imagelists_VISDA_from_list(self.src_imgs, self.src_labels, 
            root=root, transform=self.data_transforms['train'])

        target_dataset_val = Imagelists_VISDA_from_list(self.trg_val_imgs, self.trg_val_labels, 
            root=root, transform=self.data_transforms['val'])    

        target_dataset_test = Imagelists_VISDA_from_list(self.trg_test_imgs, self.trg_test_labels, 
            root=root, transform=self.data_transforms['test'], test=True)

        target_dataset_unl = Imagelists_VISDA_un_from_list(self.trg_test_imgs, self.trg_test_labels, 
            root=root, transform=self.data_transforms['val'], transform2=self.data_transforms['self'])        

        self.target_dataset = Imagelists_VISDA_Target_Labeled_from_list(self.trg_train_imgs, self.trg_train_labels, 
            root=root, ways=args.ways, trg_shots=args.trg_shots, transform=self.data_transforms['train']) 


        self.class_list = return_classlist(image_set_file_s)

        n_class = len(self.class_list)
        print("%d classes in this dataset" % n_class)    
        if args.net == 'alexnet':
            bs = 32
        else:
            bs = 24

        bs = args.ways*args.trg_shots
        nw = 12

        self.source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
            num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
            source_label=train_dataset.labels, num_per_class_src=args.src_shots))

        self.source_and_pseudo_target_loader = self.source_loader

        self.target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                        num_workers=nw, shuffle=True, drop_last=True)

        self.target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                        shuffle=False, drop_last=False)

        self.target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

        self.ways, self.src_shots, self.trg_shots = args.ways, args.src_shots, args.trg_shots
        self.root = root


        # return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list


    def update_loader(self, config_imgs, config_labels, unconfig_imgs):

        # config_imgs  = [self.trg_test_imgs[idx] for idx in config_list_imgs_idx] 
        # config_labels = config_list_labels
        # unconfig_img s = [self.trg_test_imgs[idx] for idx in unconfig_list_imgs_idx] 

        labeled_imgs = self.src_imgs.tolist()
        labels = self.src_labels.tolist()

        # breakpoint()

        labeled_imgs.extend(config_imgs)        
        labels.extend(config_labels)

        labeled_imgs = np.array(labeled_imgs)
        labels = np.array(labels)

        # breakpoint()

        labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=self.root, transform=self.data_transforms['train'])

        # self.source_and_pseudo_target_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=self.ways*(self.src_shots+self.trg_shots), 
        #     num_workers=12, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid_pseudo1(num_of_class=self.ways, 
        #         source_label=self.src_labels, target_label=config_labels, num_per_class_src=self.src_shots, num_per_class_trg=self.trg_shots))

        self.source_and_pseudo_target_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=self.ways*(self.src_shots+8), 
            num_workers=12, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid_pseudo1(num_of_class=self.ways, 
                source_label=self.src_labels, target_label=config_labels, num_per_class_src=self.src_shots, num_per_class_trg=8))


    def get_transforms(self, crop_size):

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'self': transforms.Compose([
                # transforms.Resize(256),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(crop_size),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
                RandAugmentMC(n=2, m=10),
                #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        return data_transforms







# def return_dataset_unlabeled(args, unconfig_list=None, confid_list=None):

#     base_path = './data/txt/%s' % args.dataset
#     root = './data/%s/' % args.dataset

#     if unconfig_list == None:
#         image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))
#         unconfig_list = image_set_file_unl
#         confid_list = []


#     if args.net == 'alexnet':
#         crop_size = 227
#     else:
#         crop_size = 224


#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(crop_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(crop_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(crop_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'self': transforms.Compose([
#             transforms.RandomResizedCrop(crop_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
#             #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }

#     dict_path2img = None

#     target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root, transform=data_transforms['val'],transform2=data_transforms['self'])

#     target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
#                                     transform=data_transforms['train'], dict_path2img=dict_path2img) 

#     class_list = return_classlist(image_set_file_s)

#     n_class = len(class_list)
#     print("%d classes in this dataset" % n_class)    
#     if args.net == 'alexnet':
#         bs = 32
#     else:
#         bs = 24

#     bs = args.ways*args.trg_shots

#     nw = 12

#     source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
#         num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
#         source_label=train_dataset.labels, num_per_class_src=args.src_shots))


#     target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
#                                     num_workers=nw, shuffle=True, drop_last=True)

#     target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
#                                     shuffle=True, drop_last=True)

#     target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
#                                 batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

#     return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list



def update_loaders(args, src_images, src_labels, config_trg_images, config_trg_labels, unconfig_trg_images):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    # image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    # image_set_file_unl = os.path.join(base_path, 'confid_unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if unconfig_list == None:
        image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))
        unconfig_list = image_set_file_unl
        confid_list = []


    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None


    train_dataset = Imagelists_VISDA_from_list(image_set_file_s, root=root, transform=data_transforms['train'],  dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA_from_list(image_set_file_unl, root=root, transform=data_transforms['val'], transform2=data_transforms['self'])



    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots

    nw = 12


    labeled_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
        num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
        source_label=train_dataset.labels, num_per_class_src=args.src_shots))

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return labeled_data_loader, target_loader_unl




def return_dataset_balance_self(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None


    train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root, transform=data_transforms['val'],transform2=data_transforms['self'])
    
    target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
                                    transform=data_transforms['train'], dict_path2img=dict_path2img) 

    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots

    nw = 12


    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
        num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
        source_label=train_dataset.labels, num_per_class_src=args.src_shots))


    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list



def return_dataset_balance_self_fast(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset


    if args.dataset == 'visda':
        image_set_file_s = \
            os.path.join(base_path,
                         'image_list.txt')
        
        image_set_file_t = \
            os.path.join(base_path,
                         'set1_labeled_target_images_sketch_3.txt')
        
        image_set_file_t_val = \
            os.path.join(base_path,
                         'set1_unlabeled_target_images_sketch_3.txt')

        image_set_file_unl = \
            os.path.join(base_path,
                        'set1_unlabeled_target_images_sketch_3.txt')
    else:
        image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
        image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
        image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
        image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))



    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),            
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),            
            #transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dict_path2img = None
        

    if args.dataset == 'visda':
        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root+'/validation', transform=data_transforms['val'], dict_path2img=dict_path2img)    
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root+'/validation', transform=data_transforms['test'], dict_path2img=dict_path2img)
        target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root+'/validation', transform=data_transforms['val'],transform2=data_transforms['self'])

        src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
        # breakpoint()
        src_imgs = ['train/'+src for src in src_imgs]
        trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)
        trg_train_imgs = ['validation/'+trg for trg in trg_train_imgs]

        labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
        labels = np.concatenate((src_labels, trg_train_labels))
        labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=root, transform=data_transforms['train'])

    else:
        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
        target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root, transform=data_transforms['val'],transform2=data_transforms['self'])

        src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
        trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)

        labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
        labels = np.concatenate((src_labels, trg_train_labels))
        labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=root, transform=data_transforms['train'])



    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots
    nw = 12

    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
    #     num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
    #     source_label=train_dataset.labels, num_per_class_src=args.src_shots))

    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.ways*(args.src_shots+args.trg_shots), 
            num_workers=12, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid_pseudo1(num_of_class=args.ways, 
                source_label=src_labels, target_label=trg_train_labels, 
                num_per_class_src=args.src_shots, num_per_class_trg=args.trg_shots, ways=args.ways))


    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
    #                                 num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs*2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return labeled_data_loader, target_loader_val, target_loader_test, target_loader_unl, class_list




def return_dataset_balance_self_hard(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),            
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),            
            #transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None


    train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root, transform=data_transforms['val'],transform2=data_transforms['self'])
    target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
                                    transform=data_transforms['train'], dict_path2img=dict_path2img) 

    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots

    nw = 12


    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
        num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
        source_label=train_dataset.labels, num_per_class_src=args.src_shots))


    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list




def return_dataset_balance_self_hard_fast(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset


    if args.dataset == 'visda':
        image_set_file_s = \
            os.path.join(base_path,
                         'image_list.txt')
        
        image_set_file_t = \
            os.path.join(base_path,
                         'set1_labeled_target_images_sketch_3.txt')
        
        image_set_file_t_val = \
            os.path.join(base_path,
                         'set1_unlabeled_target_images_sketch_3.txt')

        image_set_file_unl = \
            os.path.join(base_path,
                        'set1_unlabeled_target_images_sketch_3.txt')
    else:
        image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
        image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
        image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
        image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))



    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),            
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),            
            #transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dict_path2img = None
        

    if args.dataset == 'visda':
        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root+'/validation', transform=data_transforms['val'], dict_path2img=dict_path2img)    
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root+'/validation', transform=data_transforms['test'], dict_path2img=dict_path2img)
        target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root+'/validation', transform=data_transforms['val'],transform2=data_transforms['self'])

        src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
        # breakpoint()
        src_imgs = ['train/'+src for src in src_imgs]
        trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)
        trg_train_imgs = ['validation/'+trg for trg in trg_train_imgs]

        labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
        labels = np.concatenate((src_labels, trg_train_labels))
        labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=root, transform=data_transforms['train'])

    else:
        target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
        target_dataset_unl = Imagelists_VISDA_un(image_set_file_unl, root=root, transform=data_transforms['val'],transform2=data_transforms['self'])

        src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
        trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)

        labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
        labels = np.concatenate((src_labels, trg_train_labels))
        labeled_dataset = Imagelists_VISDA_from_list(labeled_imgs, labels, root=root, transform=data_transforms['train'])



    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots
    nw = 12

    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
    #     num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
    #     source_label=train_dataset.labels, num_per_class_src=args.src_shots))

    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.ways*(args.src_shots+args.trg_shots), 
            num_workers=12, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid_pseudo1(num_of_class=args.ways, 
                source_label=src_labels, target_label=trg_train_labels, 
                num_per_class_src=args.src_shots, num_per_class_trg=args.trg_shots, ways=args.ways))


    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
    #                                 num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs*2, num_workers=nw,
                                    shuffle=False, drop_last=False)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return labeled_data_loader, target_loader_val, target_loader_test, target_loader_unl, class_list






def return_dataset_balance_fast_2unl(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')

    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')

    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
        
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None

    train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)
    target_dataset_unl = Imagelists_VISDA_Twice(image_set_file_unl, root=root, transform=TransformTwice(data_transforms['val']))

    target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, 
                                    transform=data_transforms['train'], dict_path2img=dict_path2img) 


    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    nw = 12

    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
        num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
        source_label=train_dataset.labels, num_per_class_src=args.src_shots))

    # source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,
    #                                 num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list



def return_dataset_balance(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
        
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None

    training_dataset = Imagelists_VISDA_Src_Trg_Wt_Unl(image_set_file_s, image_set_file_t, image_set_file_unl, 
                                        root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)

    source_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)

    class_list = return_classlist(image_set_file_s)

    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    nw = 12
    source_loader_eval = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=nw, shuffle=False, drop_last=False)
    target_loader_eval = \
        torch.utils.data.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)),
                                    num_workers=nw, shuffle=False, drop_last=False)

    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    return training_dataset, source_loader_eval, target_loader_eval, target_loader_val, target_loader_test, class_list



def return_dataset_eval(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['test'])

    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    source_loader_eval = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=4, shuffle=False,
                                                drop_last=False)
    target_loader_eval = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=4,
                                    shuffle=False, drop_last=False)

    return source_loader_eval, target_loader_eval



def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    # image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')

    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' +args.source + '.txt')

    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=4,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list




def read_target_split(image_list):
    with open(image_list) as f:
        image_index = [int(x) for x in f.readlines()]
    return image_index


def load_pkl(path):

    with open(path, 'rb') as f:        
        data = pickle.load(f)
        tr_x, tr_y, te_x, te_y = data['TR'][0], data['TR'][1], data['TE'][0], data['TE'][1]

        # breakpoint()
        # tr_x =  tr_x.astype(np.float32)
        # te_x =  te_x.astype(np.float32)
        # tr_x = np.transpose(tr_x, (0, 3, 1, 2)).astype(np.float32)
        # te_x = np.transpose(te_x, (0, 3, 1, 2)).astype(np.float32)

        # tr_x = np.transpose(tr_x, (0, 3, 1, 2)).astype(np.float32)
        # te_x = np.transpose(te_x, (0, 3, 1, 2)).astype(np.float32)

        tr_y = tr_y.ravel().astype('int32')
        te_y = te_y.ravel().astype('int32')
    return tr_x, tr_y, te_x, te_y


def return_dataset_DIGIT(root, src, tgt, tgt_list_labeled_file, bs=256, ratio=3):

    tgt_list_labeled_file = os.path.join(root+'/'+tgt, tgt_list_labeled_file)
    tgt_list_labeled = read_target_split(tgt_list_labeled_file)


    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(36),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),        
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }        

    # breakpoint()

    src_path = root + src + '/' + src.lower() + '.pkl'
    tgt_path = root + tgt + '/' + tgt.lower() + '.pkl'
    trs_x, trs_y, tes_x, tes_y = load_pkl(src_path)
    trt_x, trt_y, tet_x, tet_y = load_pkl(tgt_path)


    tgt_list_label_np = np.asarray(tgt_list_labeled)
    tgt_list_unl_np = np.asarray(list(set(range(len(trt_y)))-set(tgt_list_labeled)))


    trt_x_train = trt_x[tgt_list_label_np]
    trt_y_train = trt_y[tgt_list_label_np]
    trt_x_unl = trt_x[tgt_list_unl_np]
    trt_y_unl = trt_y[tgt_list_unl_np]

    train_set = DomainArrayDataset([trs_x, trs_y], [trt_x_train, trt_y_train],
                                        tforms=data_transforms['train'], tformt=data_transforms['train'], ratio=ratio)
    test_set = DomainArrayDataset([tet_x, tet_y], tforms=data_transforms['test'])

    train_lorder = torch.utils.data.DataLoader(train_set, batch_size=bs,
                                                num_workers=12, shuffle=True, drop_last=False)
    test_lorder = torch.utils.data.DataLoader(test_set, batch_size=bs, 
                                    num_workers=12, shuffle=False, drop_last=False)

    return train_lorder, test_lorder



def return_dataset_DIGIT_triplet(root, src, tgt, tgt_list_labeled_file, bs=256, ratio=3):

    tgt_list_labeled_file = os.path.join(root+'/'+tgt, tgt_list_labeled_file)
    tgt_list_labeled = read_target_split(tgt_list_labeled_file)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(36),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),        
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }        


    src_path = root + src + '/' + src.lower() + '.pkl'
    tgt_path = root + tgt + '/' + tgt.lower() + '.pkl'
    trs_x, trs_y, tes_x, tes_y = load_pkl(src_path)
    trt_x, trt_y, tet_x, tet_y = load_pkl(tgt_path)
    # breakpoint()

    tgt_list_label_np = np.asarray(tgt_list_labeled)
    tgt_list_unl_np = np.asarray(list(set(range(len(trt_y)))-set(tgt_list_labeled)))


    trt_x_train = trt_x[tgt_list_label_np]
    trt_y_train = trt_y[tgt_list_label_np]
    trt_x_unl = trt_x[tgt_list_unl_np]
    trt_y_unl = trt_y[tgt_list_unl_np]

    train_set = DomainArrayDataset_Triplet([trs_x, trs_y], [trt_x_train, trt_y_train],
                                        tforms=data_transforms['train'], tformt=data_transforms['train'], ratio=ratio)
    test_set = DomainArrayDataset_Triplet([tet_x, tet_y], tforms=data_transforms['test'])

    train_lorder = torch.utils.data.DataLoader(train_set, batch_size=bs,
                                                num_workers=12, shuffle=True, drop_last=False)
    test_lorder = torch.utils.data.DataLoader(test_set, batch_size=bs, 
                                    num_workers=12, shuffle=False, drop_last=False)

    return train_lorder, test_lorder




def read_dataset_from_file(root, file, is_tgt=False):
    file_list = os.path.join(root, file)
    image_paths = []
    labels = []

    with open(file_list) as f:        
        if is_tgt:
            for ll in f.readlines():            
                label_path = ll.rstrip().split('\t')
                label, img_path = int(label_path[1]), label_path[2]
                img_path = root + '/validation/' + img_path    
                image_paths.append(img_path)
                labels.append(label)
        else:
            for ll in f.readlines():            
                label_path = ll.rstrip().split(' ')            
                img_path, label = label_path[0], int(label_path[1])                
                img_path = root + '/train/' + img_path
                image_paths.append(img_path) 
                labels.append(label) 

    return image_paths, labels


def return_dataset_VISDA_triplet(root, src_file, tgt_labeled_file, tgt_unl_file, bs=256, ratio=3):

    trs_x, trs_y = read_dataset_from_file(root, src_file, is_tgt=False)
    tet_x, tet_y = read_dataset_from_file(root, tgt_labeled_file, is_tgt=True)
    trt_x_train, trt_y_train = read_dataset_from_file(root, tgt_unl_file, is_tgt=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    
    train_set = DomainArrayDataset_Triplet([trs_x, trs_y], [trt_x_train, trt_y_train],
                    tforms=data_transforms['train'], tformt=data_transforms['train'], ratio=ratio, path_only=True)

    test_set = DomainArrayDataset_Triplet([tet_x, tet_y], tforms=data_transforms['test'], path_only=True)

    train_lorder = torch.utils.data.DataLoader(train_set, batch_size=bs,
                    num_workers=12, shuffle=True, drop_last=False)
    test_lorder = torch.utils.data.DataLoader(test_set, batch_size=bs, 
                    num_workers=12, shuffle=False, drop_last=False)

    return train_lorder, test_lorder




def return_dataset_balance_self_3u(args):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'self': transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            RandAugmentMC(n=2, m=10),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dict_path2img = None


    train_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'], dict_path2img=dict_path2img)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'], dict_path2img=dict_path2img)    
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'], dict_path2img=dict_path2img)

    target_dataset_unl = Imagelists_VISDA_un_3u(image_set_file_unl, root=root, transform=data_transforms['val'], transform2=data_transforms['self'], transform3=data_transforms['val'])

    target_dataset = Imagelists_VISDA_Target_Labeled(image_set_file_t, root=root, ways=args.ways, trg_shots=args.trg_shots,
                                    transform=data_transforms['train'], dict_path2img=dict_path2img) 

    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    bs = args.ways*args.trg_shots

    nw = 12


    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ways*args.src_shots, 
        num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler_alignedreid(num_of_class=args.ways, 
        source_label=train_dataset.labels, num_per_class_src=args.src_shots))


    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=nw,
                                    shuffle=True, drop_last=True)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs*2, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader, target_dataset, target_loader_val, target_loader_test, target_loader_unl, class_list
