## Introduction

The code is used for generating crop datasets.

The code is derived form [mmdetection](https://github.com/open-mmlab/mmdetection)

## Get Started

1. Install the environment according to the instructions of mmdetection official [website](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md).

2. Download the pretrained Hybrid Task Cascade([HTC](https://open-mmlab.oss-cn-beijing.aliyuncs.com/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth)) model from coco.

    Assume that you have already downloaded the checkpoints to the directory `checkpoints/`. 

3. ```bash
   bash run.sh
   ```


### Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{he2020multi,
 title={Multi-Domain Learning and Identity Mining for Vehicle Re-Identification},
 author={He, Shuting and Luo, Hao and Chen, Weihua and Zhang, Miao and Zhang, Yuqi and Wang, Fan and Li, Hao and Jiang, Wei},
 booktitle={Proc. CVPR Workshops},
 year={2020}
}
```

