import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp

import mscoco.dataloader_sub
from misc import torchutils, imutils
import net.resnet50_cam
import cv2
cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    # mde_predictor = net.resnet50_cam.Class_Predictor(20, 2048)
    # mde_predictor.load_state_dict(torch.load(osp.join(args.mde_weight_dir,'mde_predictor_'+str(args.mde_num_epoches) + '.pth')))

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        # mde_predictor.cuda()
        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            if os.path.exists(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy'))):
                continue
            
            print(img_name)
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            # outputs = [model.forward2(img[0].cuda(non_blocking=True),mde_predictor.classifier.weight) for img in pack['img']] # b x 20 x w x h
            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            # print(torch.stack(highres_cam, 0).shape)
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            # print(highres_cam.shape)
            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            # cv2.imshow('highres', (highres_cam[0].cpu().numpy()*255.0).astype('uint8'))
            # cv2.waitKey(0)

            # outputs0 = [model.forward(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h
            # highres_cam0 = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False) for o in outputs0]
            # highres_cam0 = torch.sum(torch.stack(highres_cam0, 0), 0)[:, 0, :size[0], :size[1]]
            # highres_cam0 = highres_cam0[valid_cat]
            # highres_cam0 /= F.adaptive_max_pool2d(highres_cam0, (1, 1)) + 1e-5

            # outputs_fc2 = [model.forward1(img[0].cuda(non_blocking=True),mde_predictor.classifier.weight) for img in pack['img']] # b x 20 x w x h
            # highres_cam_fc2 = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False) for o in outputs_fc2]
            # highres_cam_fc2 = torch.sum(torch.stack(highres_cam_fc2, 0), 0)[:, 0, :size[0], :size[1]]
            # highres_cam_fc2 = highres_cam_fc2[valid_cat]
            # highres_cam_fc2 /= F.adaptive_max_pool2d(highres_cam_fc2, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
                    # {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy(),"high_res_fc2_only":highres_cam_fc2.cpu().numpy(),"high_res0": highres_cam0.cpu().numpy(),})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(n_classes=args.num_classes)
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    # model.load_state_dict(torch.load(osp.join(args.mde_weight_dir,'res50_mde_'+str(args.mde_num_epoches) + '.pth')))
    model.eval()

    n_gpus = torch.cuda.device_count()

    # dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = mscoco.dataloader_sub.COCOClassificationDatasetMSF(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy', 
        scales=args.cam_scales,num_classes=args.num_classes,
        sub_offset=args.sub_offset, max_obj=20)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()