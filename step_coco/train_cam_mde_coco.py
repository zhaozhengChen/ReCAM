import time
import torch
import importlib
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion

cudnn.enabled = True

import mscoco.dataloader
import net.resnet50_cam
from misc import pyutils, torchutils, imutils

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

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return

def caculate_miou(confusion,name,thres,n):
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print('----------' + name + '---------')
    print({'iou': iou, 'miou': np.nanmean(iou)})
    print("threshold:", thres, 'miou:', np.nanmean(iou), "i_imgs", n)
    print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
    return np.nanmean(iou),iou

def evaluate_cam_multi_thres(args,thres_min,thres_max,epoch,cam_type):
    print('Multi Threshold evaluating.... cam_type:',cam_type)
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(n_classes=80)
    if epoch==0:
        # model.load_state_dict(torch.load('workspace_coco/coco_baseline28/res50_cam.pth'), strict=True)
        model.load_state_dict(torch.load(args.cam_weights_name))
    else:
        model.load_state_dict(torch.load(osp.join(args.mde_weight_dir,'res50_mde_'+str(epoch) + '.pth')))
        mde_predictor = net.resnet50_cam.Class_Predictor(80, 2048)
        mde_predictor.load_state_dict(torch.load(osp.join(args.mde_weight_dir,'mde_predictor_'+str(epoch) + '.pth')))
        mde_predictor.cuda()
        
    model.cuda()
    model.eval()

    # dataset = mscoco.dataloader.COCOClassificationDatasetMSF('voc12/train.txt', voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = mscoco.dataloader.COCOClassificationDatasetMSF(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy', 
        scales=args.cam_scales)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
    all_cams_dict = {}
    start_time = time.time()
    with torch.no_grad():
        for i, pack in enumerate(data_loader):
            if i>2000:
                break
            if i %100==0:
                print(i,'/',len(data_loader),'time:',time.time()-start_time)
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            if cam_type==0:
                outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h
            elif cam_type==1:
                outputs = [model.forward1(img[0].cuda(non_blocking=True),mde_predictor.classifier.weight) for img in pack['img']] # b x 20 x w x h
            elif cam_type==2:
                outputs = [model.forward2(img[0].cuda(non_blocking=True),mde_predictor.classifier.weight) for img in pack['img']] # b x 20 x w x h
            elif cam_type==3:
                outputs = [model.forward3(img[0].cuda(non_blocking=True),mde_predictor.classifier) for img in pack['img']] # b x 20 x w x h
            elif cam_type==4:
                outputs = [model.forward4(img[0].cuda(non_blocking=True),mde_predictor.classifier.weight) for img in pack['img']] # b x 20 x w x h
            elif cam_type==5:
                outputs = [model.forward5(img[0].cuda(non_blocking=True),mde_predictor.classifier) for img in pack['img']] # b x 20 x w x h
            else:
                print('Unknown cam type')
                return
            strided_up_size = imutils.get_strided_up_size(size, 16)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]
            # highres_cam /= torch.max(highres_cam) + 1e-5
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            highres_cam = highres_cam[valid_cat]
            # print(highres_cam.shape)
            highres_cam = highres_cam.cpu().numpy()
            # np.save(os.path.join(args.cam_out_dir,img_name + '.npy'), {'keys':valid_cat,'high_res':highres_cam})
            all_cams_dict[img_name] = {'keys':valid_cat,'high_res':highres_cam}
    
    dataset = mscoco.dataloader.COCOSegmentationDataset(image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        masks_path=osp.join(args.mscoco_root,'mask/train2014'),
        crop_size=512)
    
    all_cams = all_cams_dict
    max_iou = 0
    max_thres = 0
    all_miou = []
    miou_all_thres = []
    for thres in range(int(100*thres_min),int(100*thres_max)):
        thres = thres/100.0
        preds = []
        labels = []
        n_img = 0
        for i in all_cams.keys():
            highres_cam = all_cams[i]['high_res']
            cams = np.pad(highres_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres)
            keys = np.pad(all_cams[i]['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            label = dataset.get_label_by_name(i)
            labels.append(label)
            n_img += 1

        confusion = calc_semantic_segmentation_confusion(preds, labels)
        # print(confusion.shape)

        print('########### Thres:'+str(thres)+'  #########')
        overall_iou,iou = caculate_miou(confusion,'overall',thres,n_img)
        all_miou.append(overall_iou)
        miou_all_thres.append(iou)
        if overall_iou > max_iou:
            max_iou = overall_iou
            max_thres = thres
    print(args.work_space)
    print('Max overall iou:'+str(max_iou)+', thres='+str(max_thres))
    print('All miou',all_miou)
    x = np.array(miou_all_thres)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(round(x[i,j],4),end=' ')
        print()
    

def run(args):
    # evaluate_cam_multi_thres(args,0.,0.1,0,0)   
    # evaluate_cam_multi_thres(args,0.05,0.2,4,0)   
    # evaluate_cam_multi_thres(args,0.05,0.25,4,1)   
    # evaluate_cam_multi_thres(args,0.05,0.25,4,2)
    # return
    # evaluate_cam_multi_thres(args,0.15,0.2,0,0) 
    # evaluate_cam_multi_thres(args,0.1,0.4,1,2)     
    # return
    print('train_cam_mde_coco')
    model = getattr(importlib.import_module(args.cam_network), 'Net_CAM_Feature')(n_classes=80)
    param_groups = model.trainable_parameters()
    # model.load_state_dict(torch.load('./cam2_13/mde_weight/res50_mde_4.pth'), strict=True)
    model.load_state_dict(torch.load('workspace_coco/coco_baseline28/res50_cam.pth'), strict=True)
    # model.load_state_dict(torch.load('coco_sub8/res50_cam.pth'), strict=True)
    # model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model = torch.nn.DataParallel(model).cuda()

    mde_predictor = net.resnet50_cam.Class_Predictor(80, 2048)
    # mde_predictor.load_state_dict(torch.load('./cam2_6/mde_weight/mde_predictor_4.pth'))
    mde_predictor = torch.nn.DataParallel(mde_predictor).cuda()
    mde_predictor.train()

    train_dataset = mscoco.dataloader.COCOClassificationDataset(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy', 
        resize_long=(320, 640), hor_flip=True, crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.mde_batch_size,shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.mde_batch_size) * args.mde_num_epoches

    # val_dataset = mscoco.dataloader.COCOClassificationDataset(
    #     image_dir = osp.join(args.mscoco_root,'val2014/'),
    #     anno_path= osp.join(args.mscoco_root,'annotations/instances_val2014.json'),
    #     labels_path='./mscoco/val_labels.npy',crop_size=512)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.mde_batch_size,shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 0.1*args.mde_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 0.1*args.mde_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': mde_predictor.parameters(), 'lr': args.mde_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.mde_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    global_step = 0
    start_time = time.time()
    for ep in range(args.mde_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.mde_num_epoches))
        model.train()
        print('step')
        for step, pack in enumerate(train_data_loader):

            img = pack['img'].cuda()
            label = pack['label'].cuda(non_blocking=True)
            x,cam,_ = model(img)

            # print(x.shape)
            loss_cls = F.multilabel_soft_margin_loss(x, label)
            loss_mde,acc = mde_predictor(cam,label)
            loss_mde = loss_mde.mean()
            acc = acc.mean()
            loss = loss_cls + args.mde_loss_weight*loss_mde

            avg_meter.add({'loss_cls': loss_cls.item()})
            avg_meter.add({'loss_mde': loss_mde.item()})
            avg_meter.add({'acc': acc.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if (global_step-1)%100 == 0:
                timer.update_progress(global_step / max_step)

                print('step:%5d/%5d' % (global_step - 1, max_step),
                      'loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'loss_mde:%.4f' % (avg_meter.pop('loss_mde')),
                      'acc:%.4f' % (avg_meter.pop('acc')),
                      'imps:%.1f' % ((step + 1) * args.mde_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[2]['lr']),
                      'time:%ds' % (int(time.time()-start_time)), 
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        
        # validate(model, val_data_loader)
        timer.reset_stage()
        torch.save(model.module.state_dict(), osp.join(args.mde_weight_dir,'res50_mde_'+str(ep+1) + '.pth'))    
        torch.save(mde_predictor.module.state_dict(), osp.join(args.mde_weight_dir,'mde_predictor_'+str(ep+1) + '.pth'))
        evaluate_cam_multi_thres(args,0.12,0.2,ep+1,0)   
        evaluate_cam_multi_thres(args,0.18,0.3,ep+1,1)   
        evaluate_cam_multi_thres(args,0.18,0.3,ep+1,2)    
        print('Max')
    # evaluate_precision_recall(args,0,t) 
    torch.cuda.empty_cache()
