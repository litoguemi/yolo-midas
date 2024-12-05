# yolo-midas Project

## Overview
This project aims to enhance the efficiency of real-time autonomous navigation and augmented reality applications by combining object detection and depth estimation. Typically, separate networks like YOLO v3 for object detection and MiDaS for depth estimation are used, but this can be computationally expensive on edge devices with limited inference power. The solution proposed is to use a single feature extraction network, specifically ResNext101, which is the backbone of MiDaS, and train the YOLO v3 head on top of it. This approach optimizes the inference process, making it more efficient and suitable for real-time applications.

## Seting up the project

### Clone the repository

`git clone https://github.com/litoguemi/yolo-midas.git`

### Install the requirements

`pip install -r requirements.txt`

## Training

The model is trained on Construction Safety Gear Data which can be found here https://github.com/sarvan0506/EVA5-Vision-Squad/tree/Saravana/14_Construction_Safety_Dataset. If training need to done on custom datasets refer the data preparation steps mentioned in the page.

Place the data inside `data/customdata/custom.data` folder

`python3.6 main.py train --data data/customdata/custom.data --batch 8 --cache --cfg cfg/mde.conf --epochs 50 --img-size 512`

Please refer the config file `cfg/mde.cfg` to change the network configuration, freeze different branches. The model is an extension of YOLOv3 and MiDaS networks. Most of the configurations can be understood if familiar with

1. https://github.com/ultralytics/yolov3
2. https://github.com/intel-isl/MiDaS


## Inference

Download the weights from https://drive.google.com/drive/u/2/folders/11p7hhea2Y1FK_T5_P3W271V76IKigcUD and place it under `weights` folder

Place the images on which inference need to be run, inside `input` folder

`python3.6 main.py detect --source input --conf-thres 0.1 --output output --weights weights/best.pt`

The inferred images will be stored inside `output` folder

## Inference Result Sample

![result](assets/results.png)

## References
1. https://github.com/ultralytics/yolov3
2. https://github.com/intel-isl/MiDaS
3. https://sarvan0506.medium.com/yolo-v3-and-midas-from-a-single-resnext101-backbone-8ba42948bf65
4. https://github.com/sarvan0506/yolo-midas
