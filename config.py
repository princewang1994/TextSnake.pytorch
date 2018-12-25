from easydict import EasyDict

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 4

config.max_epochs = 100

config.start_epoch = 0

config.lr = 1e-5

config.cuda = True

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v