import os
import imageio
import numpy as np
from torch import multiprocessing
from pycocotools.coco import COCO
from torch.utils.data import Subset

category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

def work(process_id, infer_dataset, coco, mask_path):
    databin = infer_dataset[process_id]
    print(len(databin))
    for imgId in databin:
        curImg = coco.imgs[imgId]
        imageSize = (curImg['height'], curImg['width'])
        labelMap = np.zeros(imageSize)

        # Get annotations of the current image (may be empty)
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
        imgAnnots = coco.loadAnns(annIds)

        # Combine all annotations of this image in labelMap
        # labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
        for i in range(len(imgAnnots)):
            labelMask = coco.annToMask(imgAnnots[i]) == 1
            newLabel = imgAnnots[i]['category_id']
            labelMap[labelMask] = category_map[str(newLabel)]

        imageio.imsave(os.path.join(mask_path, str(imgId) + '.png'), labelMap.astype(np.uint8))

if __name__ == '__main__':
    annFile = '../MSCOCO/annotations/instances_train2014.json'
    mask_path = '../MSCOCO/mask/train2014'
    os.makedirs(mask_path, exist_ok=True)
    coco = COCO(annFile)
    num_workers = 8
    ids = list(coco.imgs.keys())
    print(len(ids))
    num_per_worker = (len(ids)//num_workers) + 1
    dataset = [ ids[i*num_per_worker:(i+1)*num_per_worker] for i in range(num_workers)]
    multiprocessing.spawn(work, nprocs=num_workers, args=(dataset,coco,mask_path), join=True)

    annFile = '../MSCOCO/annotations/instances_val2014.json'
    mask_path = '../MSCOCO/mask/val2014'
    os.makedirs(mask_path, exist_ok=True)
    coco = COCO(annFile)
    ids = list(coco.imgs.keys())
    print(len(ids))
    num_per_worker = (len(ids)//num_workers) + 1
    dataset = [ ids[i*num_per_worker:(i+1)*num_per_worker] for i in range(num_workers)]
    multiprocessing.spawn(work, nprocs=num_workers, args=(dataset,coco,mask_path), join=True)