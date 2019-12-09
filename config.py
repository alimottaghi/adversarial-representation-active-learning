import numpy as np


class mnist_config:
    dataset = 'mnist'
    image_size = 28 * 28
    num_label = 10

    noise_size = 100
    
    gan_lambda = 1
    cls_lambda = 1
    trd_lambda = 0.001
    adv_lambda = 0.001
    
    dis_lr = 3e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    smp_lr = 3e-3
    
    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200
    
    max_epochs = 2000

    eval_period = 500
    save_period = 10000

    data_root = 'data'
    log_root = 'logs'
    
    
class svhn_config:
    dataset = 'svhn'
    image_size = 3 * 32 * 32
    num_label = 10

    noise_size = 100
    
    gan_lambda = 1
    cls_lambda = 1
    trd_lambda = 0.001
    adv_lambda = 0.001

    dis_lr = 1e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    smp_lr = 1e-3
    min_lr = 1e-4
    
    train_batch_size = 64
    train_batch_size_2 = 64
    dev_batch_size = 200

    max_epochs = 900

    eval_period = 500
    save_period = 10000

    data_root = 'data'
    log_root = 'logs'
    

class cifar_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 10

    noise_size = 100
    
    gan_lambda = 1
    cls_lambda = 1
    trd_lambda = 0.001
    adv_lambda = 0.001

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4
    smp_lr = 6e-4
    
    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200
    
    max_epochs = 1200

    eval_period = 500
    save_period = 10000

    data_root = 'data'
    log_root = 'logs'
