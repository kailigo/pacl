# Semi-Supervised Domain Adaptation with Prototypical Alignment and Consistency Learning

This repository includes the PyTorch implementation for RCAN introduced in the following paper:

[Kai Li](http://kailigo.github.io/), [Chang Liu](https://sites.google.com/view/cliu5/home), [Handong Zhao](https://hdzhao.github.io/) [Yulun Zhang](http://yulunzhang.com/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), "Semi-Supervised Domain Adaptation with Prototypical Alignment and Consistency Learning",  [[arXiv]](https://arxiv.org/abs/1807.02758)

## Install

`pip install -r requirements.txt`

The code is written for Pytorch 0.4.0, but should work for other version
with some modifications.

## Data preparation (DomainNet)

To get data, run

`sh download_data.sh`

The images will be stored in the following way.

`./data/multi/real/category_name`,

`./data/multi/sketch/category_name`

The dataset split files are stored as follows,

`./data/txt/multi/labeled_source_images_real.txt`,

`./data/txt/multi/unlabeled_target_images_sketch_3.txt`,

`./data/txt/multi/validation_target_images_sketch_3.txt`.


## Training

The following scripts reproduce our results for the adaptation result between Real and Sketch domains from the DomainNet dataset under the 3-shot settings, with AlexNet and ResNet-34 as the backbone respectively. Other results can be obtained by changing the parameter '--source' and '--target' and '--trg_shots', which specify the source domain, target domain, and number of labeled samples from the target, respectively.

`CUDA_VISIBLE_DEVICES=0 python main.py --beta 1.0 --alpha 0.1 --threshold 0.8 --align_type proto --log_file r2s_proto_alex_num3_semi_kld_hard --kld --labeled_hard --trg_shots 3 --num 3 --net alexnet --source real --target sketch`


`CUDA_VISIBLE_DEVICES=0 python main.py --beta 1.0 --alpha 0.1 --threshold 0.8 --align_type proto --log_file r2s_proto_resnet_num3_semi_kld_hard --kld --labeled_hard --trg_shots 3 --num 3 --net resnet34 --source real --target sketch`


## Test

### Pretrained models



## Citation 

```
@misc{li2021semisupervised,
      title={Semi-Supervised Domain Adaptation with Prototypical Alignment and Consistency Learning}, 
      author={Kai Li and Chang Liu and Handong Zhao and Yulun Zhang and Yun Fu},
      year={2021},
      eprint={2104.09136},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgment 
This code is developed based on the implementation of [MME](https://github.com/VisionLearningGroup/SSDA_MME).



