import cv2
import time
import torch
import numpy as np
import os.path as osp
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion


import importlib

import voc12.dataloader
import net.resnet50_cam
from misc import pyutils, torchutils, imutils
from torch import autograd
import os

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x,_,_= model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):
    print('train_recam')
    model = getattr(importlib.import_module(args.cam_network), 'Net_CAM_Feature')()
    param_groups = model.trainable_parameters()
    model.load_state_dict(torch.load('sess/res50_cam.pth'), strict=True)
    model = torch.nn.DataParallel(model).cuda()

    recam_predictor = net.resnet50_cam.Class_Predictor(20, 2048)
    recam_predictor = torch.nn.DataParallel(recam_predictor).cuda()
    recam_predictor.train()


    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.recam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 0.1*args.recam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 0.1*args.recam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': recam_predictor.parameters(), 'lr': args.recam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.recam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    global_step = 0
    for ep in range(args.recam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.recam_num_epoches))
        model.train()

        for step, pack in enumerate(train_data_loader):

            img = pack['img'].cuda()
            label = pack['label'].cuda(non_blocking=True)
            x,cam,_ = model(img)


            loss_cls = F.multilabel_soft_margin_loss(x, label)
            loss_ce,acc = recam_predictor(cam,label)
            loss = loss_cls + args.recam_loss_weight*loss_ce

            avg_meter.add({'loss_cls': loss_cls.item()})
            avg_meter.add({'loss_ce': loss_ce.item()})
            avg_meter.add({'acc': acc.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if (global_step-1)%100 == 0:
                timer.update_progress(global_step / max_step)

                print('step:%5d/%5d' % (global_step - 1, max_step),
                      'loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'loss_ce:%.4f' % (avg_meter.pop('loss_ce')),
                      'acc:%.4f' % (avg_meter.pop('acc')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[2]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        
        validate(model, val_data_loader)
        timer.reset_stage()
        torch.save(model.module.state_dict(), osp.join(args.recam_weight_dir,'res50_recam_'+str(ep+1) + '.pth'))    
        torch.save(recam_predictor.module.state_dict(), osp.join(args.recam_weight_dir,'recam_predictor_'+str(ep+1) + '.pth'))
    torch.cuda.empty_cache()
