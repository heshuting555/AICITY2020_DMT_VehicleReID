# Multi-Domain Learning and Identity Mining for Vehicle Re-Identification

This repository contains our source code of Track2 in the NVIDIA AI City Challenge at CVPR 2020 Workshop. [Our paper](http://arxiv.org/abs/2004.10547)

## Authors

- [Shuting He](https://github.com/heshuting555)
- [Hao Luo](https://github.com/michuanhaohao)
- [Miao Zhang](https://github.com/greathope)

## Introduction

Detailed information of NVIDIA AI City Challenge 2020 can be found [here](https://www.aicitychallenge.org/).

The code is modified from [reid_strong baseline]( https://github.com/michuanhaohao/reid-strong-baseline ) and [person_reid_tiny_baseline](https://github.com/lulujianjie/person-reid-tiny-baseline).

## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/heshuting555/AICITY2020_DMT_VehicleReID.git`

3. Install dependencies:
   - [pytorch>=1.1.0](https://pytorch.org/)
   - python>=3.5
   - torchvision
   - [yacs](https://github.com/rbgirshick/yacs)
   - cv2
   
   We use cuda 9.0/python 3.6.7/torch 1.2.0/torchvision 0.4.0 for training and testing.
   
4. Prepare dataset. we have to change the first line in `AIC20_track2/AIC20_ReID/train_label.xml` as below:

   ```bash
   <?xml version="1.0" encoding="gb2312"?>
   ```

   into

   ```bash
   <?xml version="1.0" encoding="utf-8"?>
   ```

## RUN

1. If you want to get the same score as online in the AI City Challenge 2020 Track2. Use the following commands:

   ```bash
   bash run.sh
   ```

   **Note:** you can download our trained model and Distance matrix in the AICITY2020 [here](https://drive.google.com/open?id=1qmN2AUwQG37wXCwZYzqXP5G9pNXUye48)

4. If  you want to use our Multi-Domain Learning. 

   ```bash
   # you need to train a model in a Multi-Domain Datasets first.(E.g: you can add simulation datasets to aic and then test on the aic)
   
   python train.py --config_file='configs/baseline_aic_finetune.yml' MODEL.PRETRAIN_PATH "('your path for trained checkpoints')" MODEL.DEVICE_ID "('your device id')" OUTPUT_DIR "('your path to save checkpoints and logs')"
   ```

3. If you want to try our Identity Mining.

   ```bash
   # First, genereate the selected query ids
   
   python test_mining.py --config_file='configs/test_identity_mining.yml'  TEST.WEIGHT "('your path for trained checkpoints')" OUTPUT_DIR "('your path to save selected query id')" --thresh 0.49
   ```

   **Note:** The quality of the query id depends on the performance of TEST.WEIGHT.  And you can change the value of thresh to get more or less query ids.

   ```bash
   # Then,  train the model with trainset and testset(selected by the above selected query id)
   
   python train_IM.py --config_file='configs/baseline_aic.yml'
   --config_file_test='configs/test_train_IM.yml' OUTPUT_DIR "('your path to save checkpoints and logs')" MODEL.THRESH "(0.23)"
   ```

   **Note:**  you can change the value of MODEL.THRESH  which determines how many test sets added to the train sets.

4. If you want to generate crop images please refer to crop_dataset_generate  directory for detail.

5. You can visualize the result given a track2.txt result (AICITY required submission format). 

   ```bash
   python vis_txt_result.py --base_dir ('your path to the datasets') --result ('result file (txt format) path')
   ```


6. If  you want to use our baseline on public datasets (such as [VeRi](https://github.com/JDAI-CV/VeRidataset) datasets). 

   ```bash
   python train.py --config_file='configs/baseline_veri_r50.yml' MODEL.DEVICE_ID "('your device id')" OUTPUT_DIR "('your path to save checkpoints and logs')"
   ```

   

## Results (mAP/Rank1)

| Model                      | AICITY2020  |
| -------------------------- | ----------- |
| Resnet101_ibn_a (baseline) | 59.73/69.30 |
| +  Multi-Domain Learning   | 65.25/71.96 |
| +  Identity Mining         | 68.54/74.81 |
| +  Ensemble                | 73.22/80.42 |

| Backbone (baseline)        | [VeRi](https://github.com/JDAI-CV/VeRidataset) | download                                                     |
| -------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| ResNet50 (batch 48)        | 79.8/95.0                                      | [model](https://drive.google.com/open?id=1q5d3MG5iu_Sm0DXgBm2mBAtlfedo-ZiF) \| [log](https://drive.google.com/open?id=1iRkyRYUyhtv35ICpxZSK6fo6PrnQbXx1) |
| Resnet50_ibn_a (batch 48)  | 81.4/96.5                                      | [model](https://drive.google.com/open?id=14TTv8mEECkRLgtmmTMY9BPc70rhgmXcu) \| [log](https://drive.google.com/open?id=1VijL5BxYGbTKPzXGZypfmD0yTaeR3ERe) |
| Resnet101_ibn_a (batch 48) | 82.8/97.1                                      | [model](https://drive.google.com/open?id=1vr5KUdyRXPXLRtag6foDWY_WvKZ8VESr) \| [log](https://drive.google.com/open?id=1dI4GOVDvLNIqaX_MaS3hntU4yYJCWJZD) |

### Citation

If you find [our work](http://arxiv.org/abs/2004.10547) useful in your research, please consider citing:
```
@inproceedings{he2020multi,
 title={Multi-Domain Learning and Identity Mining for Vehicle Re-Identification},
 author={He, Shuting and Luo, Hao and Chen, Weihua and Zhang, Miao and Zhang, Yuqi and Wang, Fan and Li, Hao and Jiang, Wei},
 booktitle={Proc. CVPR Workshops},
 year={2020}
}
```
