
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
from PIL import Image
import torch.nn.functional as F

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list


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

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

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

        return {'name': name_str, 'img': img}

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        # label = torch.from_numpy(self.label_list[idx])
        # label = torch.nonzero(label)[:,0]
        # label = label[torch.randint(len(label),(1,))]
        # out['label'] = label

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

class VOC12ClassificationDataset_Single(VOC12ImageDataset):
    
    def __init__(self, img_name_list_path, voc12_root, 
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        # print()
        self.len = np.sum(self.label_list).astype(np.int)
        self.idx_map = np.zeros(self.len,dtype=np.int)
        self.bias = np.zeros(self.len,dtype=np.int)
        print('single_obj_data_num:',self.len)
        idx = 0
        for i in range(len(self.label_list)):
            x = np.sum(self.label_list[i])
            while x > 0:
                x = x-1
                self.idx_map[idx] = i
                self.bias[idx] = x
                idx = idx + 1
        print(idx)
        # print(self.bias[:30])
    def __getitem__(self, idx):
        if idx < len(self.img_name_list):
            out = super().__getitem__(idx)
            out['label'] = torch.from_numpy(self.label_list[idx])
        else:
            idx = idx%len(self.label_list)
            bias = self.bias[idx]
            idx = self.idx_map[idx]
            label = torch.from_numpy(self.label_list[idx])
            label = torch.nonzero(label)[:,0][bias]


            name = self.img_name_list[idx]
            name_str = decode_int_filename(name)

            mask = imageio.imread(os.path.join(self.voc12_root, 'SegmentationClassAug', name_str + '.png'))
            img0 = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
            # print(img0.dtype)
            # print(img)
            mask = np.stack([mask,mask,mask],axis=2)
            mask = (mask==0)*1 + (mask==(label+1).item())*1
            img_rand = np.random.randint(255, size=img0.shape)
            # wh = img0.shape[:2]
            # img_rand = np.stack([torch.ones(wh)*124,torch.ones(wh)*116,torch.ones(wh)*104],axis=2)
            img = (mask*img0+(1-mask)*img_rand).astype(np.uint8)

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
            out = {'name': name_str, 'img': img, 'label':F.one_hot(label, num_classes=20).type(torch.float32)}
        return out

    def __len__(self):
        print('len:',self.len + len(self.img_name_list))
        return self.len + len(self.img_name_list)

class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root, img_normal=TorchvisionNormalize(), scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))

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

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out

class VOC12SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

        self.cls_label_list = load_image_label_list_from_npy(self.img_name_list)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        # print(os.path.join(self.label_dir, name_str + '.png'))
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))

        img = np.asarray(img)

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

        return {'name': name, 'img': img, 'label': label, 'cls_label':torch.from_numpy(self.cls_label_list[idx])}

class VOC12_ours(Dataset):

    def __init__(self, img_name_list_path, voc12_root):

        self.ids = np.loadtxt(img_name_list_path, dtype=np.str)
        self.voc12_root = voc12_root
    def read_label(self, file, dtype=np.int32):
        f = Image.open(file)
        try:
            img = f.convert('P')
            img = np.array(img, dtype=dtype)
        finally:
            if hasattr(f, 'close'):
                f.close()

        if img.ndim == 2:
            return img
        elif img.shape[2] == 1:
            return img[:, :, 0]

    def get_label(self,i):
        label_path = os.path.join(self.voc12_root, 'SegmentationClassAug', self.ids[i] + '.png')
        label = self.read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        return label
    def get_label_by_name(self,i):
        label_path = os.path.join(self.voc12_root, 'SegmentationClassAug', i + '.png')
        label = self.read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        return label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return idx

class VOC12AffinityDataset(VOC12SegmentationDataset):
    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(img_name_list_path, label_dir, crop_size, voc12_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out

