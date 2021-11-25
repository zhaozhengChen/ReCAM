# ReCAM
The code of ReCAM for Anonymous CVPR 2022 submission 5021 (Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation).

## Prerequisite
- Python 3.6, PyTorch 1.9, and others in requirements.txt
## Usage
### Step 1. Download PASCAL VOC 2012 devkit.
Download PASCAL VOC 2012 devkit provided by [AdvCAM](https://github.com/jbeomlee93/AdvCAM). [Download](https://drive.google.com/file/d/1e-yprFZzOYDAehjyMVyC5en5mNq6Mjh4/view?usp=sharing)
### Step 2. Train ReCAM and generate seeds.
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --train_cam_pass True --train_recam_pass True --make_recam_pass True --eval_cam_pass True 
```
### Step 3. Train IRN and generate pseudo masks.
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 
```
### Step 4. Train semantic segmentation network.
To train DeepLab-v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). We use the ImageNet pre-trained model provided by [AdvCAM](https://github.com/jbeomlee93/AdvCAM).

## Acknowledgment
This code is borrowed from [IRN](https://github.com/jiwoon-ahn/irn) and [AdvCAM](https://github.com/jbeomlee93/AdvCAM), thanks Jiwoon and Jungbeom.
