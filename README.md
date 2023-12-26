# M-BEV: Masked BEV Perception for Robust Autonomous Driving （AAAI 2024）
### Demo for M-BEV ###
[AAAI 2024] M-BEV: This repo is the implementation of "M-BEV: Masked BEV Perception for Robust Autonomous Driving "


M-BEV is a perception framework to improve the robustness for camera-based autonomous driving methods.
We develop a novel Masked View Reconstruction (MVR) module in our M-BEV. It mimics various missing cases by randomly masking features of different camera views, then leverages the original features of these views as self-supervision, and reconstructs the masked ones with the distinct spatio-temporal context across camera views. Via such a plug-and-play MVR, our M-BEV is capable of learning the missing
views from the resting ones, and thus well generalized for robust view recovery and accurate perception in the testing.


## Preparation
This implementation is built upon [detr3d](https://github.com/WangYueFt/detr3d/blob/main/README.md) and [petrv2](https://github.com/megvii-research/PETR/edit/main/README.md). Follow their instructions to prepare for the Envirorments, besides you need to install timm and torchvision.

* Detection Data   
Follow the mmdet3d to process the nuScenes dataset (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).

* Segmentation Data  
Download Map expansion from nuScenes dataset (https://www.nuscenes.org/nuscenes#download)

* Ckpts
To verify the performance on the val set, we provide normally trained petrv2 and petrv2 trained with M-BEV[weights](https://pan.baidu.com/s/10J98exFM1nozD8cUh7zuTQ) for comparsion, 
