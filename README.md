# EAN-efficient-attention-network
[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/gbup-%E7%A8%B3%E4%BD%8F-blue.svg)

By [Zhongzhan Huang](https://github.com/dedekinds), [Senwei Liang](https://leungsamwai.github.io), [Mingfu Liang](https://github.com/wuyujack), [Wei He](https://github.com/erichhhhho) and [Haizhao Yang](https://haizhaoyang.github.io/).

The implementation of paper ''Efficient Attention Network: Accelerate Attention by Searching Where to Plug'' [[paper]](https://arxiv.org/). 

## Introduction
Efficient Attention Network (EAN) is a framework to improve the efficiency for the existing attention modules in computer vision. In EAN, we leverage the sharing mechanism [(Huang et al. 2020)](https://arxiv.org/pdf/1905.10671.pdf) to share the attention module within the backbone and search where to connect the shared attention module via reinforcement learning. 

<img src="https://github.com/gbup-group/EAN-efficient-attention-network/blob/main/image/qpt.jpg" width = "630" height = "340" align=center />


## Requirement
* Python 3.6 and [PyTorch 1.0](http://pytorch.org/)

## Implementation
Our implementation is divided in three parts. First, we pre-train a supernet. Second, we use a policy-gradient-based method to search for an optimal connection scheme from the supernet. Last, we train from scratch a network searched by the second step. 

### Pretrain a Supernet
First, we pretrain a supernet and the checkpoint is saved in NAS_ckpts. For example, we train a SGE-supernet, 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet/train_imagenet_ensemble_subset.py -a forward_config_share_sge_resnet50 -data /home/jovyan/ILSVRC2012_Data --checkpoint NAS_ckpts/ensemble_sge_train_on_subset
```
or train a DIA-supernet, 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet/train_imagenet_ensemble_subset.py -a forward_dia_fbresnet50 -data /home/jovyan/ILSVRC2012_Data --checkpoint NAS_ckpts/ensemble_dia_train_on_subset
```

### Search an Optimal Connection Scheme
Then, we search an optimal connection scheme from supernet. 


For SGE,
```
python search_imagenet/run_code_search_sge.py
```
For DIA,
```
python search_imagenet/run_code_search_dia.py
```


### Train a Network From Scratch
Last, we train from scracth the attention network with the connection scheme searched in the second step. Note that to train the attention network with the different scheme, we need to edit train_imagenet/run_codes_train_from_scratch.py
```
python train_imagenet/run_codes_train_from_scratch.py
```
The checkpoints will be save in NAS_ckpts.


