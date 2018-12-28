import os
import time
import numpy as np
import torch

import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset.total_text import TotalText
from network.loss import TextLoss
from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import BaseTransform, Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseTrainOptions
from util.visualize import visualize_detection


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def inference(model, detector, test_loader):

    model.eval()
    losses = AverageMeter()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(test_loader):

        print('Processing {}.'.format(meta['image_id']))
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
        # inference
        output = model(img)

        batch_result = detector.detect(output[0])  # (n_tcl, 3)
        for idx in range(len(batch_result)):
            img_show = img[idx].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
            visualize_detection(img_show, batch_result[idx], meta['image_id'][idx])

    print('Validation Loss: {}'.format(losses.avg))


def main():

    testset = TotalText(
        data_root='data/total-text',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet()
    load_model(model, '/data/prince/project/TextSnake/save/test2/textsnake_vgg_190.pth')

    # copy to cuda
    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True
    detector = TextDetector()

    print('Start testing TextSnake.')

    inference(model, detector, test_loader)

    print('End.')


if __name__ == "__main__":
    # parse arguments
    option = BaseTrainOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()