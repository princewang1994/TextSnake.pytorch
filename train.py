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
from network.loss import *

def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    for i, (img, score_map, geo_map, training_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry = model(img)
        loss1 = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss1.item(), img.size(0))

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            print('EAST <==> TRAIN <==> Epoch: [{0}][{1}/{2}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f})\n'.format(
                epoch, i, len(train_loader), loss=losses))


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

    criterion = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    # init or resume
    if cfg.resume and os.path.isfile(cfg.checkpoint):
        weightpath = os.path.abspath(cfg.checkpoint)
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
        checkpoint = torch.load(weightpath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))
    else:
        start_epoch = 0
    print('EAST <==> Prepare <==> Network <==> Done')

    for epoch in range(start_epoch, cfg.max_epochs):

        train(train_loader, model, criterion, scheduler, optimizer, epoch)

        if epoch % cfg.eval_iteration == 0:

            # create res_file and img_with_box
            output_txt_dir_path = predict(model, criterion, epoch)

            if hmean_ > hmean:
                is_best = True

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'is_best': is_best,
            }
            save_checkpoint(state, epoch)


if __name__ == "__main__":
    main()
