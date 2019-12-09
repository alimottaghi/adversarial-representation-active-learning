import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

import data
import config
import mnist_model

import random
import time
import os, sys
import math
import argparse
from collections import OrderedDict

import numpy as np
from utils import *

class Trainer(nn.Module):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        if args is not None:
            for k, v in args.__dict__.items():
                setattr(self.config, k, v)
        setattr(self.config, 'save_dir', '{}/{}_{}'.format(self.config.log_root, self.config.dataset, self.config.suffix))
        if hasattr(config, 'inherit'):
            setattr(self.config, 'inherit_dir', '{}/{}_{}'.format(self.config.log_root, self.config.dataset, self.config.inherit))
        
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        log_path = os.path.join(self.config.save_dir, '{}_{}_log.txt'.format(self.config.dataset, self.config.suffix))
        self.logger = open(log_path, 'a')
        
        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        self.logger.write(disp_str)
        self.logger.flush()
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        self.dis = mnist_model.Discriminative(noise_size=config.noise_size, num_label=config.num_label).cuda()
        self.gen = mnist_model.Generator(image_size=config.image_size, noise_size=config.noise_size, num_label=config.num_label).cuda()
        self.enc = mnist_model.Encoder(image_size=config.image_size, noise_size=config.noise_size, num_label=config.num_label, output_params=True).cuda()
        self.smp = mnist_model.Sampler(noise_size=config.noise_size).cuda()

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=config.dis_lr, betas=(0.5, 0.9999))
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.9999))
        self.enc_optimizer = optim.Adam(self.enc.parameters(), lr=config.enc_lr, betas=(0.0, 0.9999))
        self.smp_optimizer = optim.Adam(self.smp.parameters(), lr=config.smp_lr, betas=(0.5, 0.9999))

        self.d_criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        iter = self.load_checkpoint(self.config.save_dir)
        if iter==0 and hasattr(config, 'inherit'):
            self.load_checkpoint(self.config.inherit_dir)
            self.iter_cnt = 0

        self.labeled_loader, self.unlabeled_loader, self.unlabeled_loader2, self.dev_loader = data.get_mnist_loaders(config)

        
    def _train(self, labeled=None, vis=False, iter=0):
        config = self.config
        self.dis.train()
        self.gen.train()
        self.enc.train()
        self.smp.train()

        ##### train Dis
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = lab_images.cuda(), lab_labels.cuda()

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = unl_images.cuda()

        noise = torch.rand(unl_images.size(0), config.noise_size).cuda()
        label = torch.randint(config.num_label, (unl_images.size(0), )).cuda()
        gen_images = self.gen(noise, label)
        
        lab_logits = self.dis(lab_images)
        unl_logits = self.dis(unl_images)
        gen_logits = self.dis(gen_images.detach())

        lab_loss = config.cls_lambda * self.d_criterion(lab_logits, lab_labels)

        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        true_loss = config.gan_lambda * (-0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp)))
        fake_loss = config.gan_lambda * 0.5 * torch.mean(F.softplus(gen_logsumexp))
        unl_loss = true_loss + fake_loss
         
        d_loss = lab_loss + unl_loss

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen & Enc
        unl_images, _ = self.unlabeled_loader2.next()
        unl_images = unl_images.cuda()
        noise = torch.rand(unl_images.size(0), config.noise_size).cuda()
        label = torch.randint(config.num_label, (unl_images.size(0), )).cuda()
        gen_images = self.gen(noise, label)

        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        g_loss = torch.mean((torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)) ** 2)

        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = lab_images.cuda(), lab_labels.cuda()

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = unl_images.cuda()
        unl_pls = self.dis(unl_images)
        
        lab_mu, lab_log_sigma = self.enc(lab_images, lab_labels)
        unl_mu, unl_log_sigma = self.enc(unl_images, unl_pls)
        
        z = sample_z(unl_mu, unl_log_sigma)
        unl_images_recon = self.gen(z, unl_pls).detach()
        rec = self.mse_loss(unl_images_recon, unl_images)
        kl = -0.5 * torch.sum(1 + unl_log_sigma - unl_mu.pow(2) - unl_log_sigma.exp())
        
        vae_loss = config.trd_lambda * (rec + kl)
        
        lab_preds = self.smp(lab_mu).view(-1)
        unl_preds = self.smp(unl_mu).view(-1)

        lab_real_preds = torch.ones(lab_images.size(0)).cuda()
        unl_real_preds = torch.ones(unl_images.size(0)).cuda()
        
        s_loss = config.adv_lambda * (self.bce_loss(lab_preds, lab_real_preds) + self.bce_loss(unl_preds, unl_real_preds))
        
        ge_loss = g_loss + vae_loss + s_loss
        
        self.gen_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        ge_loss.backward()
        self.enc_optimizer.step()
        self.gen_optimizer.step()
        
        ##### train Smp
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = lab_images.cuda(), lab_labels.cuda()

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = unl_images.cuda()
        unl_pls = self.dis(unl_images)
        
        lab_mu, _ = self.enc(lab_images, lab_labels)
        unl_mu, _ = self.enc(unl_images, unl_pls)
        
        lab_preds = self.smp(lab_mu).view(-1)
        unl_preds = self.smp(unl_mu).view(-1)

        lab_real_preds = torch.ones(lab_images.size(0)).cuda()
        unl_fake_preds = torch.zeros(unl_images.size(0)).cuda()
        
        s_loss = config.adv_lambda * (self.bce_loss(lab_preds, lab_real_preds) + self.bce_loss(unl_preds, unl_fake_preds))
        
        self.smp_optimizer.zero_grad()
        s_loss.backward()
        self.smp_optimizer.step()

        monitor_dict = OrderedDict([
                       ('dis loss' , d_loss.item()),
                       ('gen & enc loss' , ge_loss.item()),
                       ('smp loss' , s_loss.item())
                   ])
                
        return monitor_dict

    def eval(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()
        self.smp.eval()

        loss, correct, cnt = 0, 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader.get_iter()):
                images = images.cuda()
                labels = labels.cuda()
                pred_prob = self.dis(images)
                loss += self.d_criterion(pred_prob, labels).item()
                cnt += 1
                correct += torch.eq(torch.max(pred_prob, 1)[1], labels).data.sum().float()
                if max_batch is not None and i >= max_batch - 1: break
        return loss / cnt, correct / (cnt * labels.shape[0])
    
    def eval2(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()
        self.smp.eval()

        lab_preds = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader.get_iter()):
                images = images.cuda()
                labels = labels.cuda()
                mu, _ = self.enc(images, labels)
                pred_prob2 = self.smp(mu)
                lab_preds.extend(pred_prob2)
                if max_batch is not None and i >= max_batch - 1: break
            lab_preds = torch.stack(lab_preds)
            lab_preds = lab_preds.view(-1)
            lab_preds *= -1
        return lab_preds
        
    def load_checkpoint(self, dir_path):
        checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
        if len(checkpoints) == 0:
            self.iter_cnt = 0
        else:
            last = max(checkpoints)
            self.iter_cnt = int(last[-11:-3])
            load_model_by_name(self, dir_path, self.iter_cnt)
        
        return self.iter_cnt

    def train(self):
        config = self.config
        iter = self.iter_cnt
        monitor = OrderedDict()

        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        while True:
            
            epoch = iter / batch_per_epoch
            if epoch >= config.max_epochs:
                break

            if iter % batch_per_epoch == 0:
                epoch_ratio = float(epoch) / float(config.max_epochs)
                self.dis_optimizer.param_groups[0]['lr'] = config.dis_lr * max(0., min(3. * (1. - epoch_ratio), 1.))
                self.gen_optimizer.param_groups[0]['lr'] = config.gen_lr * max(0., min(3. * (1. - epoch_ratio), 1.))
                self.enc_optimizer.param_groups[0]['lr'] = config.enc_lr * max(0., min(3. * (1. - epoch_ratio), 1.))
                self.smp_optimizer.param_groups[0]['lr'] = config.smp_lr * max(0., min(3. * (1. - epoch_ratio), 1.))

            iter_vals = self._train(iter=iter)

            for k, v in iter_vals.items():
                if k not  in monitor:
                    monitor[k] = 0.
                monitor[k] += v

            if iter % config.save_period == 0:
                save_model_by_name(self, iter)

            if iter % config.eval_period == 0:
                train_loss, train_accuracy = self.eval(self.labeled_loader)
                dev_loss, dev_accuracy = self.eval(self.dev_loader)

                disp_str = '#{}-{}\ttrain: {:.4f}, {:.2f}% | dev: {:.4f}, {:.2f}% '.format(
                        int(epoch), iter, train_loss, 100*train_accuracy, dev_loss, 100*dev_accuracy)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)
                disp_str +=' \n'

                monitor = OrderedDict()

                self.logger.write(disp_str)
                self.logger.flush()
                sys.stdout.write(disp_str)
                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist_trainer.py')
    parser.add_argument('--suffix', default='run0', type=str, help="Suffix added to the save images.")

    args = parser.parse_args()

    trainer = Trainer(config.mnist_config(), args)
    trainer.train()


