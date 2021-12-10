import os
import torch
import imageio
import numpy as np
from misc import imutils
from torch.utils import data
import torchvision.datasets as dset

category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class GetAffinityLabelFromIndices():
    
    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 81), np.less(segm_label_to, 81))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class COCOClassificationDataset(data.Dataset):
    def __init__(self, image_dir, anno_path, labels_path=None,resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.labels_path = labels_path
        self.category_map = category_map

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
	
        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in range(l):
                item = self.coco[i]
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(labels_path)

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 # / label_num
        return label

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)
    
    def __getitem__(self, index):
        name = self.coco.ids[index]
        name = self.coco.coco.loadImgs(name)[0]["file_name"].split('.')[0]

        img = np.asarray(self.coco[index][0])

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label':self.labels[index]}
    
    def __len__(self):
        return len(self.coco)

class COCOClassificationDatasetMSF(COCOClassificationDataset):
    def __init__(self, image_dir, anno_path, labels_path=None, img_normal=TorchvisionNormalize(), hor_flip=False,scales=(1.0,)):
        self.scales = scales
        super().__init__(image_dir, anno_path, labels_path, img_normal, hor_flip)
    
    def __getitem__(self,index):
        name = self.coco.ids[index]
        name = self.coco.coco.loadImgs(name)[0]["file_name"].split('.')[0]

        img = np.asarray(self.coco[index][0])

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": self.labels[index]}
        return out

class COCOSegmentationDataset(data.Dataset):
    def __init__(self, image_dir, anno_path, masks_path, crop_size, rescale=None, img_normal=TorchvisionNormalize(), 
                hor_flip=False,crop_method='random',read_ir_label=False):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.masks_path = masks_path
        self.category_map = category_map

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.read_ir_label = read_ir_label

        self.ids2name = {}
        for ids in self.coco.ids:
            self.ids2name[ids] = self.coco.coco.loadImgs(ids)[0]["file_name"].split('.')[0]
    
    def __getitem__(self, index):
        ids = self.coco.ids[index]
        name = self.ids2name[ids]

        img = np.asarray(self.coco[index][0])
        if self.read_ir_label:
          label = imageio.imread(os.path.join(self.masks_path, name+'.png'))
        else:
            label = imageio.imread(os.path.join(self.masks_path, str(ids) + '.png'))

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label':label}
    
    def get_label_by_id(self,ids):
        label = imageio.imread(os.path.join(self.masks_path, str(ids) + '.png'))
        return label
    
    def get_label_by_name(self,name):
        # COCO_val2014_000000159977.jpg
        label = imageio.imread(os.path.join(self.masks_path, str(int(name.split('.')[0].split('_')[-1])) + '.png'))
        return label

    def __len__(self):
        return len(self.coco)

class COCOAffinityDataset(COCOSegmentationDataset):
    def __init__(self, image_dir, anno_path, label_dir, crop_size, indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(image_dir, anno_path, label_dir, crop_size, rescale, img_normal, hor_flip, crop_method=crop_method,read_ir_label=True)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out