import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from dataset.total_text import TotalText
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import BaseTransform, Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output


def save_model(model, epoch, lr):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'textsnake_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict()
    }
    torch.save(state_dict, save_path)


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        output = model(img)
        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.viz and i < cfg.vis_num:
            visualize_network_output(output, tr_mask, tcl_mask, prefix='train_{}'.format(i))

        if i % cfg.display_freq == 0:
            print('Epoch: [ {} ][ {:03d} / {:03d} ] - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                epoch, i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(), cos_loss.item(), radii_loss.item())
            )
    if epoch % cfg.save_freq == 0 and epoch > 0:
        save_model(model, epoch, scheduler.get_lr())

    print('Training Loss: {}'.format(losses.avg))


def validation(model, valid_loader, criterion):

    model.eval()
    losses = AverageMeter()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(valid_loader):

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        output = model(img)

        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
        losses.update(loss.item())

        if cfg.viz and i < cfg.vis_num:
            visualize_network_output(output, tr_mask, tcl_mask, prefix='val_{}'.format(i))

        if i % cfg.display_freq == 0:
            print(
                'Validation: - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                    loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(),
                    cos_loss.item(), radii_loss.item())
            )
    print('Validation Loss: {}'.format(losses.avg))


def main():

    trainset = TotalText(
        data_root='data/total-text',
        ignore_list='./dataset/total_text/ignore_list.txt',
        is_training=True,
        transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )

    valset = TotalText(
        data_root='data/total-text',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet()
    # model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    print('Start training TextSnake.')

    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        train(model, train_loader, criterion, scheduler, optimizer, epoch)
        validation(model, val_loader, criterion)

    print('End.')

if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()