## MVTrans: Multi-view Perception to See Transparent Objects (ICRA2023)

[**Paper**](https://ac-rad.github.io/MVTrans/) | [**Project**](https://ac-rad.github.io/MVTrans/) | [**Video**](https://youtu.be/8Qdc_xWVp-k)

This repo contains the official implementation of the paper "MVTrans: Multi-view Perception to See Transparent Objects". 

## Introduction
Transparent object perception is a crucial skill for applications such as robot manipulation in household and laboratory settings. Existing methods utilize RGB-D or stereo inputs to handle a subset of perception tasks including depth and pose estimation. However transparent object perception remains to be an open problem. In this paper, we forgo the unreliable depth map from RGB-D sensors and extend the stereo based method. Our proposed method, MVTrans, is an end-to-end multi-view architecture with multiple perception capabilities, including depth estimation, segmentation, and pose estimation. Additionally, we establish a novel procedural photo-realistic dataset generation pipeline and create a large-scale transparent object detection dataset, Syn-TODD, which is suitable for training networks with all three modalities, RGB-D, stereo and multi-view RGB.

<img width="90%" src="model.jpg"/>

## Installation
Setup a conda environment, install required packages, and download the repo:
``` 
conda create -y --prefix ./env python=3.8
./env/bin/python -m pip install -r requirements.txt
git clone https://github.com/ac-rad/transparent-perception.git
```
Weights & Biases (wandb) is used to log and visualize training results. Please follow the [instruction](https://docs.wandb.ai/) to setup wandb. To appropriately log results to cloud, insert your wandb login key [here](https://github.com/ac-rad/transparent-perception/blob/main/net_train_multiview.py#L140) in the code. Otherwise, to log results locally, run the following command and access results at localhost:
```
wandb offline
```

## Dataset
Our synthetic transparent object detection dataset (Syn-TODD) can be downloaded at [here](https://ac-rad.github.io/MVTrans/). 

## Training
To train MVTrans from scratch, modify the data path and output directory in configuration files under `config/`, and then run:
```
./runner.sh net_train_multiview.py @config/net_config_blender_multiview_{NUM_OF_VIEW}.txt
```

## Evaluation
To run the evaluation, need to change modify the data path and output directory in configuration files under `config/`, and then run:
```
./runner.sh net_train_multiview.py @config/net_config_blender_multiview_{NUM_OF_VIEW}.txt
```
## Inference
To run the inference, launch jupyter notebook and run `inference.ipynb`.
## Citation
Please cite our paper:
```
citation{   }
```

## Reference
Our MVTrans architecture is built based on [SimNet](https://github.com/ToyotaResearchInstitute/simnet) and [ESTDepth](https://github.com/xxlong0/ESTDepth).
