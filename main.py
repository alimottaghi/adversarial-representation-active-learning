import torch
import torchvision.utils as vutils

import data
import config

import mnist_trainer
import svhn_trainer
import cifar_trainer

import argparse
import numpy as np

from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--dataset', default='mnist', type=str, help="Dataset")
    parser.add_argument('--budget', default=100, type=int, help="Budget")
    parser.add_argument('--max_iterations', default=10, type=int, help="Max Iterations")
    parser.add_argument('--max_epochs', default=100, type=int, help="Max Epochs")
    parser.add_argument('--suffix', default='run0', type=str, help="Suffix added to the save directory.")

    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        conf = config.mnist_config()
        num_examples = 60000
        Trainer = mnist_trainer.Trainer 
    elif args.dataset == 'svhn':
        conf = config.svhn_config()
        num_examples = 73257
        Trainer = svhn_trainer.Trainer 
    elif args.dataset == 'cifar':
        conf = config.cifar_config()
        num_examples = 50000
        Trainer = cifar_trainer.Trainer 
    else:
        raise NotImplementedError
    
    conf.log_root = check_folder(conf.log_root + '/{}_{}_{}'.format(args.dataset, args.budget ,args.suffix))
    conf.max_epochs = args.max_epochs
    mask = np.zeros(num_examples, dtype=np.bool)
    
    print('------------------------- Iteration 1 -------------------------')
    conf.suffix = 'iter1'
    conf.mask_file = conf.log_root + '/mask_{}_{}.npy'.format(args.dataset, conf.suffix)
    if os.path.exists(conf.mask_file):
        mask = np.load(conf.mask_file)
        print('{} loaded'.format(conf.mask_file))
    else:
        mask = querry(mask, np.random.permutation(num_examples), args.budget)
        np.save(conf.mask_file, mask)
        print('{} saved'.format(conf.mask_file))
    
    model = Trainer(conf, None)
    model.train()
    
    for i in range(args.max_iterations):
        iter = i + 2
        print('------------------------- Iteration {} -------------------------'.format(iter))
        conf.suffix = 'iter{}'.format(iter)
        conf.inherit = 'iter{}'.format(iter-1)
        conf.mask_file = conf.log_root + '/mask_{}_{}.npy'.format(args.dataset, conf.suffix)
        if os.path.exists(conf.mask_file):
            mask = np.load(conf.mask_file)
            print('{} loaded'.format(conf.mask_file))
        else:
            preds = model.eval2(model.unlabeled_loader2).cpu()
            prefs = np.argsort(preds)
            mask = querry(mask, prefs, args.budget)
            np.save(conf.mask_file, mask)
            print('{} saved'.format(conf.mask_file))

        model = Trainer(conf, None)
        model.train()
        
    