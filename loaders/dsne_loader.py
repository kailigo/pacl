from multiprocessing import cpu_count


class DomainArrayDataset(Dataset):
    """
    Domain Array Dataset, designed for digits datasets
    """
    def __init__(self, arrs=None, arrt=None, tforms=None, tformt=None, ratio=0):
        """
        Initialization of dataset
        :param arrs: source array
        :param arrt: target array
        :param tforms: transformers for source array
        :param tformt: transformers for target array
        :param ratio: negative/positive ratio
        """
        assert arrs is not None or arrt is not None, "One of src array or tgt array should not be None"

        self.arrs = arrs
        self.use_src = False if arrs is None else True

        self.arrt = arrt
        self.use_tgt = False if arrt is None else True

        self.tforms = tforms
        self.tformt = tformt

        self.ratio = ratio

        # breakpoint()

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
        for ids, ys in enumerate(self.arrs[1]):
            for idt, yt in enumerate(self.arrt[1]):
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
            im = nd.array(im, dtype='float32')

            if self.tforms is not None:
                im = self.tforms(im)

            return im, l
        elif self.use_tgt and not self.use_src:
            im, l = self.arrt[0][idx], self.arrt[1][idx]
            im = nd.array(im, dtype='float32')
            if self.tformt is not None:
                im = self.tformt(im)

            return im, l
        else:
            [ids, ys, idt, yt, lc] = self.pairs[idx]
            ims, ls = self.arrs[0][ids], self.arrs[1][ids]
            imt, lt = self.arrt[0][idt], self.arrt[1][idt]

            ims = nd.array(ims, dtype='float32')
            imt = nd.array(imt, dtype='float32')

            assert ys == ls
            assert yt == lt

            if self.tforms is not None:
                ims = self.tforms(ims)


            if self.tformt is not None:
                imt = self.tformt(imt)


            return ims, ls, imt, lt, lc

    def __len__(self):
        return len(self.pairs)





class DomainArrayDataset_Triplet(Dataset):
    def __init__(self, arrs=None, arrt=None, tforms=None, tformt=None, ratio=0):
        
        assert arrs is not None or arrt is not None, "One of src array or tgt array should not be None"

        self.arrs = arrs
        self.use_src = False if arrs is None else True

        self.arrt = arrt
        self.use_tgt = False if arrt is None else True

        self.tforms = tforms
        self.tformt = tformt

        self.ratio = ratio

        # breakpoint()
        self.all_class = np.unique(label_t)

        if self.use_src and self.use_tgt:
            self.cls2sample_s, self.cls2sample_t = self.create_dict_cls2sample()
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
        

        for c in all_class:
            idx = np.where(label_s==c)[0]
            cls2sample_s[c] = idx

            idx = np.where(label_t==c)[0]
            cls2sample_t[c] = idx

        return cls2sample_s, cls2sample_t


    def __getitem__(self, idx):

        if self.use_src and not self.use_tgt:
            im, l = self.arrs[0][idx], self.arrs[1][idx]
            im = nd.array(im, dtype='float32')

            if self.tforms is not None:
                im = self.tforms(im)

            return im, l
        elif self.use_tgt and not self.use_src:
            im, l = self.arrt[0][idx], self.arrt[1][idx]
            im = nd.array(im, dtype='float32')
            if self.tformt is not None:
                im = self.tformt(im)

            return im, l
        else:
            self.cls2sample_s 
            self.cls2sample_t

            [ids, ys, idt, yt, lc] = self.pairs[idx]
            ims, ls = self.arrs[0][ids], self.arrs[1][ids]
            imt, lt = self.arrt[0][idt], self.arrt[1][idt]

            ims = nd.array(ims, dtype='float32')
            imt = nd.array(imt, dtype='float32')

            assert ys == ls
            assert yt == lt

            if self.tforms is not None:
                ims = self.tforms(ims)

            if self.tformt is not None:
                imt = self.tformt(imt)

            return ims, ls, imt, lt, lc

    def __len__(self):
        return len(self.pairs)






class DSNE_loader(object):
	def __init__(self, dataset, src, trg, bs):

		self.dataset = dataset		
		self.transform = transform
        self.create_loader(transform)
        self.bs = bs

    def create_loader(self):
        cpus = cpu_count()

        if self.dataset = 'digits':
            trs_set, trt_set, tes_set, tet_set = self.create_digits_datasets(train_tforms, eval_tforms)
        elif self.dataset = 'office':
            trs_set, trt_set, tes_set, tet_set = self.create_office_datasets(train_tforms, eval_tforms)
        elif self.dataset = 'visda':
            trs_set, trt_set, tes_set, tet_set = self.create_visda_datasets(train_tforms, eval_tforms)
        else:
            raise NotImplementedError

        self.train_src_loader = DataLoader(trs_set, self.args.bs, shuffle=True, num_workers=cpus)
        self.test_src_loader = DataLoader(tes_set, self.args.bs, shuffle=False, num_workers=cpus)
        self.test_tgt_loader = DataLoader(tet_set, self.args.bs, shuffle=False, num_workers=cpus)



    def load_digits_cfg(self):
        cfg = load_json(self.args.cfg)
        cfg = split_digits_train_test(cfg, self.args.src.upper(), self.args.tgt.upper(), 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['TR']
        trt = cfg[self.args.tgt.upper()]['TR']
        tes = cfg[self.args.src.upper()]['TE']
        tet = cfg[self.args.tgt.upper()]['TE']

        return trs, trt, tes, tet

    def load_office_cfg(self):
        cfg = load_json(self.args.cfg)
        cfg = split_office_train_test(cfg, 1, self.args.seed)

        trs = cfg[self.args.src.upper()]['SRC-TR']
        trt = cfg[self.args.tgt.upper()]['TGT-TR']
        tes = cfg[self.args.src.upper()]['TGT-TE']
        tet = cfg[self.args.tgt.upper()]['TGT-TE']

        return trs, trt, tes, tet

    def load_visda_cfg(self):
        cfg = load_json(self.args.cfg)

        trs = cfg['SRC']['TRAIN']
        trt = cfg['TGT']['TRAIN']
        tes = cfg['SRC']['TRAIN']
        tet = cfg['TGT']['TEST']
        self.label_dict = cfg['Label']

        return trs, trt, tes, tet

    def create_digits_datasets(self, train_tforms, eval_tforms):
        """
        Create digits datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        trs, trt, tes, tet = self.load_digits_cfg()

        # breakpoint()
        if self.args.aug_tgt_only:
            trs_set = DomainArrayDataset(trs, tforms=train_tforms)
        else:
            trs_set = DomainArrayDataset(trs, tforms=eval_tforms)
        trt_set = DomainArrayDataset(trt, tforms=train_tforms)
        tes_set = DomainArrayDataset(tes, tforms=eval_tforms)
        tet_set = DomainArrayDataset(tet, tforms=eval_tforms)

        return trs_set, trt_set, tes_set, tet_set



        

    def create_office_datasets(self, train_tforms, eval_tforms):
        """
        Create Office datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        trs, trt, tes, tet = self.load_office_cfg()

        if self.args.aug_tgt_only:
            trs_set = DomainArrayDataset(trs, tforms=train_tforms)
        else:
            trs_set = DomainArrayDataset(trs, tforms=eval_tforms)
        trt_set = DomainFolderDataset(trt, tforms=train_tforms)
        tes_set = DomainFolderDataset(tes, tforms=eval_tforms)
        tet_set = DomainFolderDataset(tet, tforms=eval_tforms)

        return trs_set, trt_set, tes_set, tet_set

    def create_visda_datasets(self, train_tforms, eval_tforms):
        """
        Create VisDA17 datasets
        :param train_tforms: training transformers
        :param eval_tforms: evaluation transformers
        :return:
            trs_set: training source set
            trt_set: training target set
            tes_set: testing source set
            tet_set: testing target set
        """
        # Read config
        trs, trt, tes, tet = self.load_visda_cfg()

        if self.args.aug_tgt_only:
            trs_set = DomainArrayDataset(trs, tforms=train_tforms)
        else:
            trs_set = DomainArrayDataset(trs, tforms=eval_tforms)
        trt_set = DomainRecDataset(trt, tforms=train_tforms)
        tes_set = DomainRecDataset(tes, tforms=eval_tforms)
        tet_set = DomainRecDataset(tet, tforms=eval_tforms)

        return trs_set, trt_set, tes_set, tet_set