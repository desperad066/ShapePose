import torch
import torch.nn as nn
import numpy as np


def iou(predict, target, eps=1e-6):
    # print('iou predict',predict.size())
    # print('iou target',target.size())
    # print(predict.ndimension())
    # print(range(predict.ndimension())[1:])
    
    dims = tuple(range(predict.ndimension())[1:])
    # print(dims)
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target):
    return 1 - iou(predict, target)

# [0] because the predicts is tuple
def multiview_iou_loss(predicts, targets_a, targets_b):

    # print('multi predict1', predicts[0].size())
    # print('multi predict2', predicts[1].size())
    # print('multi predict3', predicts[2].size())
    # print('multi predict4', predicts[3].size())

    # print('targets_a',targets_a.size())
    # print('targets_b',targets_b.size())

    # print('alpha',predicts[0][:, 3])
    # print('alpha',predicts[0][:, 3].size())

    # =============================================================
    # print('alpha non zero?', predicts[0][0, 3].nonzero().size(0))
    # print('alpha non zero?', targets_a[0, 3].nonzero().size(0))
    # print('alpha non zero?', targets_b[0, 3].nonzero().size(0))

    # print('alpha non zero?', predicts[0][1, 3].nonzero().size(0))
    # print('alpha non zero?', targets_a[1, 3].nonzero().size(0))
    # print('alpha non zero?', targets_b[1, 3].nonzero().size(0))

    # print('the non-zero number', predicts[0][0, 3].sum())
    # print('the non-zero number', targets_a[0, 3].sum())
    # print('the non-zero number', targets_b[0, 3].sum())

    # print('the non-zero number', predicts[0][1, 3].sum())
    # print('the non-zero number', targets_a[1, 3].sum())
    # print('the non-zero number', targets_b[1, 3].sum())

    loss = (iou_loss(predicts[0][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[1][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[2][:, 3], targets_b[:, 3]) + \
            iou_loss(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss
def singleview_iou_loss(predicts, targets_a):
    loss = iou_loss(predicts[:, 3], targets_a[:, 3])
    return loss