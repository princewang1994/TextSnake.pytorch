import time
import os
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler

from util import AverageMeter
from dataset.total_text import TotalText
from network.textnet import TextNet
from augmentation import BaseTransform
from config import config as cfg, update_config
from network.loss import TextLoss
from option import BaseTrainOptions

def save_model(model, epoch, lr):
    save_path = os.path.join(cfg.save_dir, cfg.exp_name, 'textsnake_{}_{}.pth'.format(model.backbone_name, epoch))
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

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cfg.cuda is not None:
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = \
                img.cuda(), train_mask.cuda(), tr_mask.cuda(), tcl_mask.cuda(), radius_map.cuda(), sin_map.cuda(), cos_map.cuda()

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

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map) in enumerate(valid_loader):

        if cfg.cuda is not None:
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = \
                img.cuda(), train_mask.cuda(), tr_mask.cuda(), tcl_mask.cuda(), radius_map.cuda(), sin_map.cuda(), cos_map.cuda()

        output = model(img)
        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)

        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

        losses.update(loss.item())

        if i % cfg.display_freq == 0:
            print(
                'Validation: - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                    loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(),
                    cos_loss.item(), radii_loss.item())
            )
    print('Validation Loss: {}'.format(losses.avg))

def main():

    transform = BaseTransform(
        size=cfg.input_size, mean=cfg.means, std=cfg.stds
    )
    trainset = TotalText(
        data_root='data/total-text',
        ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    valset = TotalText(
        data_root='data/total-text',
        ignore_list=None,
        is_training=False,
        transform=transform
    )

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet()
    # model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model = model.cuda()
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
    from util import mkdirs

    # parse arguments
    option = BaseTrainOptions()
    args = option.parse()
    update_config(cfg, args)

    print('==========Options============')
    print(cfg)
    print('=============End=============')

    mkdirs(cfg.vis_dir)
    mkdirs(cfg.save_dir)
    # main
    main()