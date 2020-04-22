# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                #print('using right sampler and loss')
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet_center':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           triplet(feat, target)[0] + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + \
                           triplet(feat, target)[0] + \
                           cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


