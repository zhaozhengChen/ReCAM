# ReCAM
The code of ReCAM for Anonymous CVPR 2022 submission 5021 (Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation).

## Prerequisite
- Python 3.6, PyTorch 1.9, and others in environment.yml
- You can create the environment from environment.yml file
```
conda env create -f environment.yml
```
## Usage
### Step 1. Prepare dataset.
- Download PASCAL VOC 2012 devkit from [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). 
- You need to specify the path ('voc12_root') of your downloaded devkit in the following steps.
### Step 2. Train ReCAM and generate seeds.
- Please specify a workspace to save the model and logs.
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --train_cam_pass True --train_recam_pass True --make_recam_pass True --eval_cam_pass True 
```
### Step 3. Train IRN and generate pseudo masks.
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 
```
### Step 4. Train semantic segmentation network.
To train DeepLab-v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). 
We use the [ImageNet pre-trained model](https://drive.google.com/file/d/14soMKDnIZ_crXQTlol9sNHVPozcQQpMn/view?usp=sharing) for DeepLabV2 provided by [AdvCAM](https://github.com/jbeomlee93/AdvCAM).
Please replace the groundtruth masks with generated pseudo masks.

## Acknowledgment
This code is borrowed from [IRN](https://github.com/jiwoon-ahn/irn) and [AdvCAM](https://github.com/jbeomlee93/AdvCAM), thanks Jiwoon and Jungbeom.
