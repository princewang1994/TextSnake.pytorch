from easydict import EasyDict
import torch

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 4

config.max_epoch = 200

config.start_epoch = 0

config.lr = 1e-4

config.cuda = True

config.vis_num = 3

config.n_disk = 15


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
