import os
import time
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from util import AverageMeter
from dataset.total_text import TotalText
from network.textnet import TextNet
from augmentation import BaseTransform
from config import config as cfg
from network.loss import TextLoss


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

        losses.update(loss.data[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [ {} ][ {} / {} ] - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                epoch, i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(), cos_loss.item(), radii_loss.item())
            )

def main():

    transform = BaseTransform(
        size=512, mean=0.5, std=0.5
    )
    trainset = TotalText(
        data_root='/home/prince/ext_data/dataset/test-detection/total-text',
        ignore_list='/data/prince/project/TextSnake/ignore_list.txt',
        is_training=True,
        transform=transform
    )
    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # Model
    model = TextNet()
    # model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model = model.cuda()
    cudnn.benchmark = True

    criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    print('Start training TextSnake.')

    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        train(model, train_loader, criterion, scheduler, optimizer, epoch)

    print('End.')

if __name__ == "__main__":
    main()
