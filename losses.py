import numpy as np

from torch import nn
import torch as t
from torch.autograd import Variable
import lib.array_tool as at
from torch.nn import functional as F
from config import opt
from lib.bbox_tools import bbox_iou
from lib.array_tool import tonumpy
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum().float()  # ignore gt_label==-1 for rpn_loss
    return loc_loss

class RPNLoss(nn.Module):
    def __init__(self):
        super(RPNLoss, self).__init__()
        self.rpn_sigma = opt.rpn_sigma

    def forward(self, gt_rpn_loc,gt_rpn_label,  rpn_locs, rpn_scores):
        # Since batch size is one, convert variables to singular form
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]


        # ------------------ RPN losses -------------------#

        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        return [rpn_loc_loss, rpn_cls_loss]

class ROILoss(nn.Module):
    def __init__(self):
        super(ROILoss, self).__init__()
        self.roi_sigma = opt.roi_sigma
    def forward(self,gt_roi_loc, gt_roi_label,roi_cls_loc, roi_score):
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), at.totensor(gt_roi_label).long()]

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        return [roi_loc_loss,roi_cls_loss]

class RelationNetworksLoss(nn.Module):
    def __init__(self):
        super(RelationNetworksLoss, self).__init__()

    def forward(self, gt_bboxes, gt_labels, nms_scores, sorted_labels, sorted_cls_bboxes):
        if nms_scores is None:
            return [1.]
        sorted_score, prob_argsort = t.sort(nms_scores, descending=True)
        sorted_cls_bboxes = sorted_cls_bboxes[prob_argsort]
        sorted_labels = sorted_labels[prob_argsort]
        sorted_labels = tonumpy(sorted_labels)
        gt_labels = tonumpy(gt_labels)

        nms_gt = t.zeros_like(sorted_score)

        eps = 1e-8

        iou = bbox_iou(tonumpy(gt_bboxes[0]), tonumpy(sorted_cls_bboxes))
        for gt_idx in range(len(iou)):
            accept_iou = np.reshape(np.argwhere(iou[gt_idx] > 0.5),-1)
            accept_label = np.reshape(np.argwhere(sorted_labels[accept_iou] == gt_labels[0][gt_idx]),-1)

            if not(len(accept_label)==0):
                nms_gt[accept_iou[accept_label[0]]] = 1.

        loss = nms_gt * (sorted_score+ eps).log() + (1 - nms_gt) * (1-sorted_score + eps).log()
        loss = -loss.mean()
        return [loss]
