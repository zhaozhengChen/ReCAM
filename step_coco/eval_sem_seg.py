import numpy as np
import os.path as osp
import mscoco.dataloader
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

def run(args):
    dataset = mscoco.dataloader.COCOSegmentationDataset(image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        masks_path=osp.join(args.mscoco_root,'mask/train2014'),
        crop_size=512)
    preds = []
    labels = []
    n_img = 0
    num = len(dataset)
    for i, pack in enumerate(dataset):
        if i%1000==0:
            print(i,'/',num)
        img_name = pack['name'].split('.')[0]
        cls_file = img_name+'.png'
        cls_labels = imageio.imread(osp.join(args.sem_seg_out_dir, cls_file)).astype(np.uint8)
        preds.append(cls_labels.copy())
        label = dataset.get_label_by_name(img_name)
        labels.append(label)
        n_img += 1
    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    print("total images", n_img)
    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})
