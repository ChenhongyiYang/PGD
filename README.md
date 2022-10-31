# Prediction-Guided Distillation 

PyTorch implementation of our ECCV 2022 paper: [Prediction-Guided Distillation for Dense Object Detection](https://arxiv.org/pdf/2203.05469.pdf)

## Requirements

- Our codebase is built on top of [MMDetection](https://github.com/open-mmlab/mmdetection), which can be installed following the offcial instuctions.
- We used pytorch pre-trained [ResNets](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) for training.
- Please follow the MMdetection offcial instuction to set up COCO dataset. 
- Please download the [CrowdHuman](https://www.crowdhuman.org/) and set up the dataset by running this [script](https://github.com/ChenhongyiYang/PGD/blob/main/crowd_code/create_crowd_anno.py).

## Usage

#### Set up datasets and pre-trained models 

```shell
mkdir data
ln -s path_to_coco data/coco
ln -s path_to_crowdhuman data/crowdhuman 
ln -s path_to_pretrainedModel data/pretrain_models 
```

#### COCO Experiments 

```shell
# ------------------------------------
#    Here we use ATSS as an example
# ------------------------------------

# Training and testing teacher model
zsh tools/dist_train.sh work_configs/detectors/atss_r101_3x_ms.py 8
zsh tools/dist_test.sh work_configs/detectors/atss_r101_3x_ms.py work_dirs/atss_r101_3x_ms/latest.pth 8

# Training and testing student model 
zsh tools/dist_train.sh work_configs/detectors/atss_r50_1x.py 8
zsh tools/dist_test.sh work_configs/detectors/atss_r50_1x.py work_dirs/atss_r50_1x/latest.pth 8

# Training and testing PGD model
zsh tools/dist_train.sh work_configs/pgd_atss_r101_r50_1x.py 8
zsh tools/dist_test.sh work_configs/pgd_atss_r101_r50_1x.py work_dirs/pgd_atss_r101_r50_1x/latest.pth 8
```

#### CrowdHuman Experiments

```shell
# Training teacher, conducting KD, and evalauation
zsh tools/run_crowdhuman.sh
```

## Model Zoo

#### COCO

|  Detector  |             Setting             |     mAP     |                            Config                            |
| :--------: | :-----------------------------: | :---------: | :----------------------------------------------------------: |
|    FCOS    | Teacher (r101, 3x, multi-scale) |    43.1     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/fcos_r101_3x_ms.py) |
|     -      | Student (r50, 1x, single-scale) |    38.2     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/fcos_r50_1x.py) |
|     -      |   PGD (r50, 1x, single-scale)   | 42.5 (+4.3) | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/pgd_fcos_r101_r50_1x.py) |
| AutoAssign | Teacher (r101, 3x, multi-scale) |    44.8     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/autoassign_r101_3x_ms.py) |
|     -      | Student (r50, 1x, single-scale) |    40.6     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/autoassign_r50_1x.py) |
|     -      |   PGD (r50, 1x, single-scale)   | 43.8 (+3.1) | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/pgd_autoassign_r101_r50_1x.py) |
|    ATSS    | Teacher (r101, 3x, multi-scale) |    45.5     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/atss_r101_3x_ms.py) |
|     -      | Student (r50, 1x, single-scale) |    39.6     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/atss_r50_1x.py) |
|     -      |   PGD (r50, 1x, single-scale)   | 44.2 (+4.6) | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/pgd_atss_r101_r50_1x.py) |
|    GFL     | Teacher (r101, 3x, multi-scale) |    45.8     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/gfl_r101_3x_ms.py) |
|     -      | Student (r50, 1x, single-scale) |    40.2     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/gfl_r50_1x.py) |
|     -      |   PGD (r50, 1x, single-scale)   | 43.8 (+3.6) | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/pgd_gfl_r101_r50_1x.py) |
|    DDOD    | Teacher (r101, 3x, multi-scale) |    46.6     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/ddod_r101_3x_ms.py) |
|     -      | Student (r50, 1x, single-scale) |    42.0     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/detectors/ddod_r50_1x.py) |
|     -      |   PGD (r50, 1x, single-scale)   | 45.4 (+3.4) | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/pgd_ddod_r101_r50_1x.py) |

#### CrowdHuman

| Detector |                Setting                |    MR ↓     |    AP ↑     |    JI ↑     |                            Config                            |
| :------: | :-----------------------------------: | :---------: | :---------: | :---------: | :----------------------------------------------------------: |
|   DDOD   | Teacher (r101, 36 epoch, multi-scale) |    41.4     |    90.2     |    81.4     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/det_crowdhuman/ddod_r101.py) |
|    -     | Student (r50, 12 epoch, single-scale) |    46.0     |    88.0     |    79.0     | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/det_crowdhuman/ddod_r50.py) |
|    -     |   PGD (r50, 12 epoch, single-scale)   | 42.8 (-3.2) | 90.0 (+2.0) | 80.7 (+1.7) | [config](https://github.com/ChenhongyiYang/PGD/blob/main/work_configs/pgd_ddod_crowdhuman_r101_r50.py) |

## Ciation

```
@article{yang2022predictionguided,
  title={{Prediction-Guided Distillation for Dense Object Detection}},
  author={Yang, Chenhongyi and Ochal, Mateusz and Storkey, Amos and Crowley, Elliot J},
  journal={ECCV 2022},
  year={2022}
}
```

## Acknowledgement 

We thank [FGD](https://github.com/yzd-v/FGD) and [DDOD](https://github.com/zehuichen123/DDOD) for their code base. 
