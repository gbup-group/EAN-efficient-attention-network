# EAN-efficient-attention-network
[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/gbup-%E7%A8%B3%E4%BD%8F-blue.svg)

By [Zhongzhan Huang](https://github.com/dedekinds), [Senwei Liang](https://leungsamwai.github.io), [Mingfu Liang](https://github.com/wuyujack), [Wei He](https://github.com/erichhhhho) and [Haizhao Yang](https://haizhaoyang.github.io/).

The implementation of paper ''Efficient Attention Network: Accelerate Attention by Searching Where to Plug'' [[paper]](https://arxiv.org/). 

## Introduction
Efficient Attention Network~(EAN) is a framework to improve the efficiency for the existing attention modules in computer vision. In EAN, we leverage the sharing mechanism (Huang et al. 2020) to share the attention module within the backbone and search where to connect the shared attention module via reinforcement learning. 

## Requirement
* Python 3.6 and [PyTorch 1.0](http://pytorch.org/)
