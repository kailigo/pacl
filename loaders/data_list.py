import numpy as np
import os
import os.path
from PIL import Image
import torch
import random

from pdb import set_trace as breakpoint



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]

    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)

    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list



class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, dict_path2img=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)



class Imagelists_VISDA_from_list(object):
    def __init__(self, image_list, label_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, dict_path2img=None):

        imgs, labels = image_list, label_list

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)






class Imagelists_VISDA_un(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, transform2=None,target_transform=None,test=False, dict_path2img=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform2 is not None:
            img2 = self.transform2(img)
            return img1, img2, target

        if not self.test:
            return img1, target
        else:
            return img1, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)




class Imagelists_VISDA_un_from_list(object):
    def __init__(self, image_list, label_list, root="./data/multi/",
                 transform=None, transform2=None,target_transform=None, test=False, dict_path2img=None):    
        # imgs, labels = make_dataset_fromlist(image_list)
        imgs, labels = image_list, label_list

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform2 is not None:
            img2 = self.transform2(img)
            return img1, img2, target

        if not self.test:
            return img1, target
        else:
            return img1, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)




class Imagelists_VISDA_Twice(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, dict_path2img=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)



class DomainArrayDataset(object):
    def __init__(self, arrs=None, arrt=None, tforms=None, tformt=None, ratio=0):

        assert arrs is not None or arrt is not None, "One of src array or tgt array should not be None"

        self.arrs = arrs
        self.use_src = False if arrs is None else True

        self.arrt = arrt
        self.use_tgt = False if arrt is None else True

        self.tforms = tforms
        self.tformt = tformt
        self.ratio = ratio

        if self.use_src and self.use_tgt:
            self.pairs = self._create_pairs()
        elif self.use_src and not self.use_tgt:
            self.pairs = list(range(len(self.arrs[0])))
        elif not self.use_src and self.use_tgt:
            self.pairs = list(range(len(self.arrs[0])))
        else:
            sys.exit("Need to input one source")

    def _create_pairs(self):
        """
        Create pairs for array
        :return:
        """
        pos_pairs, neg_pairs = [], []

        for idt, yt in enumerate(self.arrt[1]):
            for ids, ys in enumerate(self.arrs[1]):
                if ys == yt:
                    pos_pairs.append([ids, ys, idt, yt, 1])
                else:
                    neg_pairs.append([ids, ys, idt, yt, 0])

        if self.ratio > 0:
            random.shuffle(neg_pairs)
            pairs = pos_pairs + neg_pairs[: self.ratio * len(pos_pairs)]
        else:
            pairs = pos_pairs + neg_pairs


        random.shuffle(pairs)

        return pairs


    def __getitem__(self, idx):

        if self.use_src and not self.use_tgt:
            im, l = self.arrs[0][idx], self.arrs[1][idx]        

            if self.tforms is not None:
                im = self.tforms(im)

            return im, l
        elif self.use_tgt and not self.use_src:
            im, l = self.arrt[0][idx], self.arrt[1][idx]

            if self.tformt is not None:
                im = self.tformt(im)

            return im, l
        else:
            [ids, ys, idt, yt, lc] = self.pairs[idx]
            ims, ls = self.arrs[0][ids], self.arrs[1][ids]
            imt, lt = self.arrt[0][idt], self.arrt[1][idt]

            assert ys == ls
            assert yt == lt

            if self.tforms is not None:
                ims = self.tforms(ims)

            if self.tformt is not None:
                imt = self.tformt(imt)

            return ims, ls, imt, lt, lc

    def __len__(self):
        return len(self.pairs)



class DomainArrayDataset_Triplet(object):
    def __init__(self, arrs=None, arrt=None, tforms=None, tformt=None, ratio=0, path_only=False):

        assert arrs is not None or arrt is not None, "One of src array or tgt array should not be None"

        self.arrs = arrs
        self.use_src = False if arrs is None else True

        self.arrt = arrt
        self.use_tgt = False if arrt is None else True

        self.tforms = tforms
        self.tformt = tformt

        self.ratio = ratio
        self.path_only = path_only
        self.loader = pil_loader

        # breakpoint()

        self.all_class = np.unique(arrs[1])
        self.nc = len(self.all_class)

        if self.use_src and self.use_tgt:
            self.cls2sample_s, self.cls2sample_t = self.create_dict_cls2sample()
            self.pairs = list(range(len(self.arrs[0])))
        elif self.use_src and not self.use_tgt:
            self.pairs = list(range(len(self.arrs[0])))
        elif not self.use_src and self.use_tgt:
            self.pairs = list(range(len(self.arrs[0])))
        else:
            sys.exit("Need to input one source")


    def create_dict_cls2sample(self):        
        cls2sample_s, cls2sample_t = {}, {}
        label_s = self.arrs[1]    
        label_t = self.arrt[1]

        for c in self.all_class:
            idx = np.where(label_s==c)[0]
            cls2sample_s[c] = idx
            idx = np.where(label_t==c)[0]
            cls2sample_t[c] = idx
        return cls2sample_s, cls2sample_t


    def __getitem__(self, idx):

        # print(idx)

        if self.use_src and not self.use_tgt:
            im, l = self.arrs[0][idx], self.arrs[1][idx]        
            if self.path_only:
                im = self.loader(im)
            if self.tforms is not None:
                im = self.tforms(im)                
            return im, l
        elif self.use_tgt and not self.use_src:
            im, l = self.arrt[0][idx], self.arrt[1][idx]
            if self.path_only:
                im = self.loader(im)
            if self.tformt is not None:
                im = self.tformt(im)
            return im, l
        else:            
            c = idx % self.nc
            idx_t = np.random.choice(self.cls2sample_t[c])            
            idx_ss = np.random.choice(self.cls2sample_s[c], size=2, replace=False)
            imt, ims1, ims2 = self.arrt[0][idx_t], self.arrs[0][idx_ss[0]], self.arrs[0][idx_ss[1]]            

            if self.path_only:
                ims1 = self.loader(ims1)
                ims2 = self.loader(ims2)
                imt = self.loader(imt)


            if self.tforms is not None:
                ims1 = self.tforms(ims1)
                ims2 = self.tforms(ims2)

            if self.tformt is not None:
                imt = self.tformt(imt)

            return ims1, c, ims2, c, imt, c 

    def __len__(self):
        if self.use_tgt and self.use_src:
            return 2400000
            # int(len(self.arrs[0]) * len(self.arrt[1]) /10)
        else:
            return len(self.pairs)
        


class Imagelists_VISDA_Src_Trg_Wt_Unl(object):
    def __init__(self, image_list_src, image_list_trg, image_list_unl, root="./data/multi/",
                 transform=None, target_transform=None, test=False, ways=10, 
                 src_shots=10, trg_shots=2, unl_shots=3, dict_path2img=None):
        src_imgs, src_labels = make_dataset_fromlist(image_list_src)
        trg_imgs, trg_labels = make_dataset_fromlist(image_list_trg)
        unl_imgs, unl_labels = make_dataset_fromlist(image_list_unl)
        
        self.src_imgs = src_imgs
        self.src_labels = src_labels
        self.trg_imgs = trg_imgs
        self.trg_labels = trg_labels
        self.unl_imgs = unl_imgs
        self.unl_labels = unl_labels

        self.ways = ways
        self.src_shots = src_shots
        self.trg_shots = trg_shots
        self.unl_shots = unl_shots

        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

        self.classes = np.unique(self.src_labels)


    def __getitem__(self, index):        
        select_labels = torch.LongTensor(self.ways*self.src_shots)
        selected_classes = np.random.choice(list(self.classes), self.ways, False)

        batch_src_imgs = torch.Tensor()
        batch_src_label = torch.zeros(self.ways*self.src_shots).long()
        batch_trg_imgs = torch.Tensor()
        batch_trg_label = torch.zeros(self.ways*self.trg_shots).long()        
        batch_unl_imgs = torch.Tensor()
        batch_unl_label = torch.zeros(self.ways*self.unl_shots).long()

        # breakpoint()

        for i in range(self.ways):
            idx = (self.src_labels==selected_classes[i]).nonzero()[0]            
            select_instances = np.random.choice(idx, self.src_shots, False)       
            for j in range(self.src_shots):
                src_img = self.src_imgs[select_instances[j]]
                path = os.path.join(self.root, src_img)                
                img = self.loader(path)

                if self.transform is not None:
                    img = self.transform(img)
                batch_src_imgs = torch.cat((batch_src_imgs, img.unsqueeze(0)))
                batch_src_label[i*self.src_shots+j] = selected_classes[i].item()

            idx = (self.trg_labels==selected_classes[i]).nonzero()[0]            
            select_instances = np.random.choice(idx, self.trg_shots, False)            
            for j in range(self.trg_shots):
                trg_img = self.trg_imgs[select_instances[j]]
                path = os.path.join(self.root, trg_img)                
                img = self.loader(path)
                # label = selected_classes[i]
                if self.transform is not None:
                    img = self.transform(img)                    
                batch_trg_imgs = torch.cat((batch_trg_imgs, img.unsqueeze(0)))
                batch_trg_label[i*self.trg_shots+j] = selected_classes[i].item()


            idx = (self.unl_labels==selected_classes[i]).nonzero()[0]            
            select_instances = np.random.choice(idx, self.unl_shots, False)            
            for j in range(self.unl_shots):
                unl_img = self.unl_imgs[select_instances[j]]
                path = os.path.join(self.root, unl_img)                
                img = self.loader(path)
                # label = selected_classes[i]
                if self.transform is not None:
                    img = self.transform(img)                    
                batch_unl_imgs = torch.cat((batch_unl_imgs, img.unsqueeze(0)))
                batch_unl_label[i*self.unl_shots+j] = selected_classes[i].item()

        return batch_src_imgs, batch_src_label, batch_trg_imgs, batch_trg_label, batch_unl_imgs, batch_unl_label


    def __len__(self):
        return len(self.src_imgs)






class Imagelists_VISDA_Target_Labeled():
    def __init__(self, image_list_trg, root="./data/multi/",
                 transform=None, target_transform=None, test=False, ways=10, 
                 trg_shots=2, dict_path2img=None):

        trg_imgs_path, trg_labels = make_dataset_fromlist(image_list_trg)

        self.trg_imgs_path = trg_imgs_path        
        self.trg_labels = trg_labels

        self.ways = ways
        self.trg_shots = trg_shots        

        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root

        self.trg_imgs = []        
        for img in self.trg_imgs_path:
            path = os.path.join(self.root, img)                
            img = self.loader(path)

            self.trg_imgs.append(img)


    def getitem(self, sampled_classes):        
        # select_labels = sampled_classes
        selected_classes = sampled_classes

        batch_trg_imgs = torch.Tensor()
        batch_trg_label = torch.zeros(self.ways*self.trg_shots).long()        

        for i in range(self.ways):
            idx = (self.trg_labels==selected_classes[i]).nonzero()[0]            
            select_instances = np.random.choice(idx, self.trg_shots, False)            
            for j in range(self.trg_shots):
                img = self.trg_imgs[select_instances[j]]
                # path = os.path.join(self.root, trg_img)                
                # img = self.loader(path)

                if self.transform is not None:
                    img = self.transform(img)                    
                batch_trg_imgs = torch.cat((batch_trg_imgs, img.unsqueeze(0)))
                batch_trg_label[i*self.trg_shots+j] = selected_classes[i].item()

        return batch_trg_imgs, batch_trg_label




class Imagelists_VISDA_Target_Labeled_from_list():
    def __init__(self, image_list, label_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, ways=10, 
                 trg_shots=2, dict_path2img=None):


        trg_imgs_path, trg_labels = image_list, label_list

        # trg_imgs_path, trg_labels = make_dataset_fromlist(image_list_trg)

        self.trg_imgs_path = trg_imgs_path        
        self.trg_labels = trg_labels

        self.ways = ways
        self.trg_shots = trg_shots        

        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root

        self.trg_imgs = []        
        for img in self.trg_imgs_path:
            path = os.path.join(self.root, img)                
            img = self.loader(path)

            self.trg_imgs.append(img)


    def getitem(self, sampled_classes):        
        # select_labels = sampled_classes
        selected_classes = sampled_classes

        batch_trg_imgs = torch.Tensor()
        batch_trg_label = torch.zeros(self.ways*self.trg_shots).long()        

        for i in range(self.ways):
            idx = (self.trg_labels==selected_classes[i]).nonzero()[0]            
            select_instances = np.random.choice(idx, self.trg_shots, False)            
            for j in range(self.trg_shots):
                img = self.trg_imgs[select_instances[j]]
                # path = os.path.join(self.root, trg_img)                
                # img = self.loader(path)

                if self.transform is not None:
                    img = self.transform(img)                    
                batch_trg_imgs = torch.cat((batch_trg_imgs, img.unsqueeze(0)))
                batch_trg_label[i*self.trg_shots+j] = selected_classes[i].item()

        return batch_trg_imgs, batch_trg_label







class Imagelists_VISDA_Target_Labeled_self():
    def __init__(self, image_list_trg, root="./data/multi/",
                 transform=None, target_transform=None, test=False, ways=10, 
                 trg_shots=2, dict_path2img=None):

        trg_imgs_path, trg_labels = make_dataset_fromlist(image_list_trg)

        self.trg_imgs_path = trg_imgs_path        
        self.trg_labels = trg_labels

        self.ways = ways
        self.trg_shots = trg_shots        

        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root

        self.trg_imgs = []        
        for img in self.trg_imgs_path:
            path = os.path.join(self.root, img)                
            img = self.loader(path)

            self.trg_imgs.append(img)


    def getitem(self, sampled_classes):        
        # select_labels = sampled_classes
        selected_classes = sampled_classes

        batch_trg_imgs = torch.Tensor()
        batch_trg_label = torch.zeros(self.ways*self.trg_shots*2).long()        

        for i in range(self.ways):
            idx = (self.trg_labels==selected_classes[i]).nonzero()[0]            
            select_instances = np.random.choice(idx, self.trg_shots, False)            
            for j in range(self.trg_shots):
                img = self.trg_imgs[select_instances[j]]
                img2 = img
                # path = os.path.join(self.root, trg_img)                
                # img = self.loader(path)

                if self.transform is not None:
                    img = self.transform(img)    
                    img2 = self.transform(img2)                 
                batch_trg_imgs = torch.cat((batch_trg_imgs, img.unsqueeze(0),img2.unsqueeze(0)))
                batch_trg_label[i*self.trg_shots+j:i*self.trg_shots+j+1] = selected_classes[i].item()

        return batch_trg_imgs, batch_trg_label


class Imagelists_VISDA_un_3u(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, transform2=None, transform3=None, target_transform=None,test=False, dict_path2img=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3

        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform2 is not None:
            img2 = self.transform2(img)
            img3 = self.transform3(img)
            return img1, img2, img3, target

        if not self.test:
            return img1, target
        else:
            return img1, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
