import os
import time
import numpy as np
import torch

import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset.total_text import TotalText
from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.misc import to_device
from util.option import BaseOptions
from util.visualize import visualize_detection


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def inference(model, detector, test_loader):

    model.eval()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(test_loader):

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
        # inference
        output = model(img)

        for idx in range(img.size(0)):
            print('detect {} images: {}.'.format(idx, meta['image_id'][idx]))

            tr_pred = output[idx, 0:2].softmax(dim=0).data.cpu().numpy()
            tcl_pred = output[idx, 2:4].softmax(dim=0).data.cpu().numpy()
            sin_pred = output[idx, 4].data.cpu().numpy()
            cos_pred = output[idx, 5].data.cpu().numpy()
            radii_pred = output[idx, 6].data.cpu().numpy()

            batch_result = detector.detect(tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred)  # (n_tcl, 3)

            # visualization
            img_show = img[idx].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
            visualize_detection(img_show, tr_pred[1], tcl_pred[1], batch_result[idx], '{}_{}'.format(i, meta['image_id'][idx]))


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
    model_path = os.path.join(cfg.save_dir, cfg.exp_name, \
              'textsnake_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    load_model(model, model_path)

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
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()