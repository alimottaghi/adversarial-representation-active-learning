# Adversarial Representation Active Learning

Pytoch implementation of ['Adversarial Representation Active Learning'](https://arxiv.org/pdf/1912.09720.pdf).

by Ali Mottaghi and Serena Yeung.


## Abstract
Active learning aims to develop label-efficient algorithms by querying the most informative samples to be labeled by an oracle. The design of efficient training methods that require fewer labels is an important research direction  which allows a more effective use of computational and human resources for labeling and training the deep neural networks. In this work, we demonstrate how we can use the recent advances in deep generative models, to outperform the state-of-the-art in achieving the highest classification accuracy using as few labels as possible. Unlike previous approaches, our approach uses not only labeled images to train the classifier, but also unlabeled images and generated images for co-training the whole model. Our experiments show that the proposed method significantly outperforms existing approaches in active learning on a wide range of datasets (MNIST, CIFAR-10, SVHN, CelebA, and ImageNet). 


## Requirements
1. Python 3
2. Pytorch 1
3. Torchvision 0.3
4. NVIDIA GPU with CUDA CuDNN


## Experiments
The code can simply be run using
```bash
python main.py --dataset mnist --budget 10
```
which starts an experiment on MNIST dataset with the labeling budget 10 for each iteration.


## Contact
If there are any questions or concerns feel free to send a message at mottaghi@stanford.edu
