import time
import math
import torch.utils.model_zoo as model_zoo
import six

from torch.nn import functional as F
from losses import ROILoss, RPNLoss, RelationNetworksLoss
from lib.nms import non_maximum_suppression
from collections import namedtuple
from string import Template
import lib.array_tool as at
from config import opt
from data.dataset import preprocess, VGGpreprocess
from lib.bbox_tools import loc2bbox

import torch as t
from torch.autograd import Function

from lib.roi_cupy import kernel_backward, kernel_forward
from lib.creator_tool import ProposalCreator, ProposalTargetCreator, AnchorTargetCreator
from lib.relation_tool import PositionalEmbedding, RankEmbedding
from torchvision.models import vgg16_bn,squeezenet1_1




import torch
import torch.nn as nn
import numpy as np
import cupy as cp


Stream = namedtuple('Stream', ['ptr'])

@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K

class SqueezeFRCN(nn.Module):
    feat_stride = 16  # downsample 16x for output of convolution squeeze
    def __init__(self, num_classes):
        super(SqueezeFRCN, self).__init__()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.n_class = num_classes +1
        self.training = False

        model = squeezenet1_1(pretrained=True)
        self.feature_extractor = model.features

        # freeze
        for layer in self.feature_extractor[:5]:
            for p in layer.parameters():
                p.requires_grad = False

        self.rpn = RegionProposalNetwork(in_channels=512, mid_channels=512, feat_stride=self.feat_stride)
        self.roi_head = RoIHead(n_class=self.n_class, roi_size=7, spatial_scale=(1. / self.feat_stride),
                                in_channels=512, fc_features=512, n_relations=0)

        self.proposal_target_creator = ProposalTargetCreator()
        self.anchor_target_creator = AnchorTargetCreator()

        self.roiLoss = ROILoss()
        self.rpnLoss = RPNLoss()

    def forward(self,inputs,scale = 1.):
        if self.training:
            img_batch, bboxes, labels, _ = inputs
        else:
            img_batch = inputs

        _, _, H, W = img_batch.shape
        img_size = (H, W)
        start = time.time()
        features = self.feature_extractor(img_batch)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)
        if self.training:
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(bboxes[0]),
                anchor,
                img_size)
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                rois,
                at.tonumpy(bboxes[0]),
                at.tonumpy(labels[0]),
                self.loc_normalize_mean,
                self.loc_normalize_std)
            sample_roi_index = t.zeros(len(sample_roi))

            roi_cls_loc, roi_score, appearance_features = self.roi_head(features, sample_roi, sample_roi_index)

            return gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores, \
                   sample_roi, roi_cls_loc, roi_score, appearance_features, img_size, labels, bboxes
        else:
            roi_cls_loc, roi_score, appearance_features = self.roi_head(features, rois, roi_indices)

            return roi_cls_loc, roi_score, rois, roi_indices, appearance_features, img_size
    def get_loss(self,inputs,isLearnNMS):
        gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores, \
        sample_roi, roi_cls_loc, roi_score, appearance_features, img_size, labels, bboxes = self(inputs)
        if(isLearnNMS):
            rpn_loss = self.rpnLoss(gt_rpn_loc,gt_rpn_label,  rpn_locs, rpn_scores)
            roi_loss = self.roiLoss(gt_roi_loc, gt_roi_label,roi_cls_loc, roi_score)
            nms_scores, sorted_labels, sorted_cls_bboxes = self.duplicate_remover(sample_roi, roi_cls_loc, roi_score,
                                                                                  appearance_features, img_size)
            nms_loss = self.nmsLoss(bboxes, labels,nms_scores, sorted_labels, sorted_cls_bboxes)
            losses = rpn_loss+roi_loss+nms_loss
            losses = [sum(losses)]+losses
            return losses
        else:
            rpn_loss = self.rpnLoss(gt_rpn_loc, gt_rpn_label, rpn_locs, rpn_scores)
            roi_loss = self.roiLoss(gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score)
            losses = rpn_loss + roi_loss
            losses = [sum(losses)]+losses
            return losses
    def predict(self, imgs, sizes=None, visualize=False):
        if visualize:
            self.use_preset(isTraining=False, preset='visualize')
            prepared_imgs = list()
            for img in imgs:
                size = img.shape[1:]
                img = VGGpreprocess(at.tonumpy(img))
                prepared_imgs.append(img)
        else:
            self.use_preset(isTraining=False, preset='evaluate')
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()
        for img in prepared_imgs:
            img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
            size = img.shape[2:]
            scale = np.array(1.)
            roi_cls_loc, roi_scores, rois, _,_ ,_ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data

            roi = at.totensor(rois)

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def use_preset(self,isTraining,preset='visualize'):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        self.training=isTraining
    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if(opt.use_adam):
            optimizer = t.optim.Adam(params)
        else:
            optimizer = t.optim.SGD(params,momentum = 0.9)
        return optimizer
class VGGFRCN(nn.Module):
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16
    def __init__(self, num_classes):
        super(VGGFRCN, self).__init__()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.n_class = num_classes+1
        self.training = False
        model = vgg16_bn(pretrained=True)
        self.feature_extractor = model.features[:43]
        # freeze top4 conv
        for layer in self.feature_extractor[:14]:
            for p in layer.parameters():
                p.requires_grad = False

        classifier = model.classifier
        del classifier[6]
        del classifier[5]
        del classifier[2]
        classifier = nn.Sequential(*classifier)

        self.rpn = RegionProposalNetwork(in_channels=512, mid_channels=512, feat_stride=self.feat_stride)

        self.roi_head = RoIHead(n_class=self.n_class, roi_size=7, spatial_scale=(1. / self.feat_stride), n_relations=0,
                                in_channels=512, fc_features=4096, classifier = classifier)

        self.proposal_target_creator = ProposalTargetCreator()
        self.anchor_target_creator = AnchorTargetCreator()

        self.roiLoss = ROILoss()
        self.rpnLoss = RPNLoss()
        self.freeze_bn()
    def forward(self,inputs, scale=1.):
        if self.training:
            img_batch, bboxes, labels, _ = inputs
        else:
            img_batch = inputs

        _, _, H, W = img_batch.shape
        img_size = (H, W)

        features = self.feature_extractor(img_batch)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)

        if self.training:
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(bboxes[0]),
                anchor,
                img_size)
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                rois,
                at.tonumpy(bboxes[0]),
                at.tonumpy(labels[0]),
                self.loc_normalize_mean,
                self.loc_normalize_std)
            sample_roi_index = t.zeros(len(sample_roi))

            roi_cls_loc, roi_score, appearance_features = self.roi_head(features, sample_roi, sample_roi_index)

            return gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores, \
                   sample_roi, roi_cls_loc, roi_score, appearance_features, img_size, labels, bboxes

        else:
            roi_cls_loc, roi_score, appearance_features = self.roi_head(features, rois, roi_indices)

            return roi_cls_loc, roi_score, rois, roi_indices, appearance_features, img_size

    def get_loss(self,inputs,isLearnNMS):
        gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores, \
        sample_roi, roi_cls_loc, roi_score, appearance_features, img_size, labels, bboxes = self(inputs)
        if(isLearnNMS):
            rpn_loss = self.rpnLoss(gt_rpn_loc,gt_rpn_label,  rpn_locs, rpn_scores)
            roi_loss = self.roiLoss(gt_roi_loc, gt_roi_label,roi_cls_loc, roi_score)
            nms_scores, sorted_labels, sorted_cls_bboxes = self.duplicate_remover(sample_roi, roi_cls_loc, roi_score,
                                                                                  appearance_features, img_size)
            nms_loss = self.nmsLoss(bboxes, labels,nms_scores, sorted_labels, sorted_cls_bboxes)
            losses = rpn_loss+roi_loss+nms_loss
            losses = [sum(losses)]+losses
            return losses
        else:
            rpn_loss = self.rpnLoss(gt_rpn_loc, gt_rpn_label, rpn_locs, rpn_scores)
            roi_loss = self.roiLoss(gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score)
            losses = rpn_loss + roi_loss
            losses = [sum(losses)]+losses
            return losses
    def predict(self, imgs, sizes=None, visualize=False):
        if visualize:
            self.use_preset(isTraining=False, preset='visualize')
            prepared_imgs = list()
            for img in imgs:
                size = img.shape[1:]
                img = VGGpreprocess(at.tonumpy(img))
                prepared_imgs.append(img)
        else:
            self.use_preset(isTraining=False, preset='evaluate')
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()
        for img in prepared_imgs:
            img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
            size = img.shape[2:]
            scale = np.array(1.)
            roi_cls_loc, roi_scores, rois, _,_ ,_ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data

            roi = at.totensor(rois)

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def use_preset(self,isTraining,preset='visualize'):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        self.training=isTraining
    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if(opt.use_adam):
            optimizer = t.optim.Adam(params)
        else:
            optimizer = t.optim.SGD(params,momentum = 0.9)
        return optimizer

# class ResFRCN(nn.Module):
#     feat_stride = 16  # downsample 32x for output of convolution resnet
#     def __init__(self, num_classes, block, layers):
#         self.training=False
#         self.inplanes = 64
#         self.loc_normalize_mean = (0., 0., 0., 0.)
#         self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
#         self.n_class = num_classes+1
#
#         super(ResFRCN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         if block == BasicBlock:
#             fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels,
#                          self.layer4[layers[3]-1].conv2.out_channels]
#             self.conv2 = nn.Conv2d(self.layer4[layers[3]-1].conv2.out_channels, 512, kernel_size=1, stride=1, bias=False)
#         elif block == Bottleneck:
#             fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels,
#                          self.layer4[layers[3]-1].conv3.out_channels]
#             self.conv2 = nn.Conv2d(self.layer4[layers[3]-1].conv3.out_channels, 512, kernel_size=1, stride=1, bias=False)
#
#         #self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],feature_size = 512)
#
#         self.rpn = RegionProposalNetwork(in_channels=512,mid_channels=512,feat_stride = self.feat_stride)
#         self.roi_head = RoIHead(n_class = num_classes+1,roi_size=7,spatial_scale=(1. / self.feat_stride),
#                                 in_channels=512,fc_features = 1024, n_relations= 16)
#         self.duplicate_remover = DuplicationRemovalNetwork(n_relations=16,appearance_feature_dim=1024,
#                                                            num_classes=num_classes)
#         self.proposal_target_creator = ProposalTargetCreator()
#         self.anchor_target_creator = AnchorTargetCreator()
#
#         self.roiLoss = ROILoss()
#         self.rpnLoss = RPNLoss()
#         self.nmsLoss = RelationNetworksLoss()
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
#         self.freeze_bn()
#     def use_preset(self,isTraining,preset='visualize'):
#         if preset == 'visualize':
#             self.nms_thresh = 0.3
#             self.score_thresh = 0.7
#         elif preset == 'evaluate':
#             self.nms_thresh = 0.3
#             self.score_thresh = 0.5
#         self.training=isTraining
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#         return nn.Sequential(*layers)
#     def freeze_bn(self):
#         '''Freeze BatchNorm layers.'''
#         for layer in self.modules():
#             if isinstance(layer, nn.BatchNorm2d):
#                 layer.eval()
#     def forward(self, inputs, scale=1.):
#         if self.training:
#             img_batch, bboxes, labels, _ = inputs
#         else:
#             img_batch = inputs
#
#         _, _, H, W = img_batch.shape
#         img_size = (H, W)
#         x = self.conv1(img_batch)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#
#         #features = self.fpn([x2, x3, x4])
#         features = self.conv2(x4)
#         rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features,img_size,scale)
#
#         if self.training:
#             gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
#                 at.tonumpy(bboxes[0]),
#                 anchor,
#                 img_size)
#             sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
#                 rois,
#                 at.tonumpy(bboxes[0]),
#                 at.tonumpy(labels[0]),
#                 self.loc_normalize_mean,
#                 self.loc_normalize_std)
#             sample_roi_index = t.zeros(len(sample_roi))
#
#             roi_cls_loc, roi_score, appearance_features = self.roi_head(features, sample_roi, sample_roi_index)
#
#             return gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores,\
#                    sample_roi, roi_cls_loc, roi_score, appearance_features, img_size,labels, bboxes
#
#         else:
#             roi_cls_loc, roi_score, appearance_features = self.roi_head(features, rois, roi_indices)
#
#             return roi_cls_loc,roi_score, rois, roi_indices, appearance_features, img_size
#
#     def _suppress(self, raw_cls_bbox, raw_prob):
#         bbox = list()
#         label = list()
#         score = list()
#         # skip cls_id = 0 because it is the background class
#         for l in range(1, self.n_class):
#             cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
#             prob_l = raw_prob[:, l]
#             mask = prob_l > self.score_thresh
#             cls_bbox_l = cls_bbox_l[mask]
#             prob_l = prob_l[mask]
#             keep = non_maximum_suppression(
#                 cp.array(cls_bbox_l), self.nms_thresh, prob_l)
#             keep = cp.asnumpy(keep)
#             bbox.append(cls_bbox_l[keep])
#             # The labels are in [0, self.n_class - 2].
#             label.append((l - 1) * np.ones((len(keep),)))
#             score.append(prob_l[keep])
#         bbox = np.concatenate(bbox, axis=0).astype(np.float32)
#         label = np.concatenate(label, axis=0).astype(np.int32)
#         score = np.concatenate(score, axis=0).astype(np.float32)
#         return bbox, label, score
#     def predict(self, imgs, sizes=None, visualize=False):
#         if visualize:
#             self.use_preset(isTraining=False, preset='visualize')
#             prepared_imgs = list()
#             sizes = list()
#             for img in imgs:
#                 size = img.shape[1:]
#                 img = preprocess(at.tonumpy(img))
#                 prepared_imgs.append(img)
#                 sizes.append(size)
#         else:
#             self.use_preset(isTraining=False, preset='evaluate')
#             prepared_imgs = imgs
#         bboxes = list()
#         labels = list()
#         scores = list()
#         for img, size in zip(prepared_imgs, sizes):
#             img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
#             scale = img.shape[3] / size[1]
#             roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
#             # We are assuming that batch size is 1.
#             roi_score = roi_scores.data
#             roi_cls_loc = roi_cls_loc.data
#             if visualize:
#                 roi = at.totensor(rois) / scale
#             else:
#                 roi = at.totensor(rois) / scale.cuda().float()
#
#             # Convert predictions to bounding boxes in image coordinates.
#             # Bounding boxes are scaled to the scale of the input images.
#             mean = t.Tensor(self.loc_normalize_mean).cuda(). \
#                 repeat(self.n_class)[None]
#             std = t.Tensor(self.loc_normalize_std).cuda(). \
#                 repeat(self.n_class)[None]
#
#             roi_cls_loc = (roi_cls_loc * std + mean)
#             roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
#             roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
#             cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
#                                 at.tonumpy(roi_cls_loc).reshape((-1, 4)))
#             cls_bbox = at.totensor(cls_bbox)
#             cls_bbox = cls_bbox.view(-1, self.n_class * 4)
#             # clip bounding box
#             cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
#             cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
#
#             prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1))
#
#             raw_cls_bbox = at.tonumpy(cls_bbox)
#             raw_prob = at.tonumpy(prob)
#
#             bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
#             bboxes.append(bbox)
#             labels.append(label)
#             scores.append(score)
#
#         # self.use_preset('evaluate')
#         # self.train()
#         return bboxes, labels, scores
#
#     def get_loss(self,inputs,isLearnNMS):
#         gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores, \
#         sample_roi, roi_cls_loc, roi_score, appearance_features, img_size, labels, bboxes = self(inputs)
#         if(isLearnNMS):
#             rpn_loss = self.rpnLoss(gt_rpn_loc,gt_rpn_label,  rpn_locs, rpn_scores)
#             roi_loss = self.roiLoss(gt_roi_loc, gt_roi_label,roi_cls_loc, roi_score)
#             nms_scores, sorted_labels, sorted_cls_bboxes = self.duplicate_remover(sample_roi, roi_cls_loc, roi_score,
#                                                                                   appearance_features, img_size)
#             nms_loss = self.nmsLoss(bboxes, labels,nms_scores, sorted_labels, sorted_cls_bboxes)
#             losses = rpn_loss+roi_loss+nms_loss
#             losses = [sum(losses)]+losses
#             return losses
#         else:
#             rpn_loss = self.rpnLoss(gt_rpn_loc, gt_rpn_label, rpn_locs, rpn_scores)
#             roi_loss = self.roiLoss(gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score)
#             losses = rpn_loss + roi_loss
#             losses = [sum(losses)]+losses
#             return losses

class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=256, mid_channels=256, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=32,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = self.generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = self._enumerate_shifted_anchor_torch(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_fg_scores = \
            rpn_scores.view(n, hh, ww, n_anchor, 2)[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    def generate_anchor_base(self,base_size=16, ratios=[0.5, 1, 2],
                             anchor_scales=[8, 16, 32]):
        """Generate anchor base windows by enumerating aspect ratio and scales.

        Generate anchors that are scaled and modified to the given aspect ratios.
        Area of a scaled anchor is preserved when modifying to the given aspect
        ratio.

        :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
        function.
        The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
        generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

        For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
        the width and the height of the base window will be stretched by :math:`8`.
        For modifying the anchor to the given aspect ratio,
        the height is halved and the width is doubled.

        Args:
            base_size (number): The width and the height of the reference window.
            ratios (list of floats): This is ratios of width to height of
                the anchors.
            anchor_scales (list of numbers): This is areas of anchors.
                Those areas will be the product of the square of an element in
                :obj:`anchor_scales` and the original area of the reference
                window.

        Returns:
            ~numpy.ndarray:
            An array of shape :math:`(R, 4)`.
            Each element is a set of coordinates of a bounding box.
            The second axis corresponds to
            :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

        """
        py = base_size / 2.
        px = base_size / 2.

        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                               dtype=np.float32)
        for i in six.moves.range(len(ratios)):
            for j in six.moves.range(len(anchor_scales)):
                h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
                w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = py - h / 2.
                anchor_base[index, 1] = px - w / 2.
                anchor_base[index, 2] = py + h / 2.
                anchor_base[index, 3] = px + w / 2.
        return anchor_base

    def _enumerate_shifted_anchor_torch(self,anchor_base, feat_stride, height, width):
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # return (K*A, 4)

        # !TODO: add support for torch.CudaTensor
        # xp = cuda.get_array_module(anchor_base)
        shift_y = t.arange(0, height * feat_stride, feat_stride)
        shift_x = t.arange(0, width * feat_stride, feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                          shift_y.ravel(), shift_x.ravel()), axis=1)

        A = anchor_base.shape[0]
        K = shift.shape[0]
        anchor = anchor_base.reshape((1, A, 4)) + \
                 shift.reshape((1, K, 4)).transpose((1, 0, 2))
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        return anchor

class RoI(Function):
    """
    NOTE：only CUDA-compatible
    """

    def __init__(self, outh, outw, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

    def forward(self, x, rois):
        # NOTE: MAKE SURE input is contiguous too
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = B, C, H, W = x.size() ## 1, 128, heights/32, width/32
        self.N = N = rois.size(0) ## 128
        output = t.zeros(N, C, self.outh, self.outw).cuda() ## 128,128,7,7
        self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
        self.rois = rois
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale, C, H, W,
                self.outh, self.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                        block=(CUDA_NUM_THREADS, 1, 1),
                        grid=(GET_BLOCKS(output.numel()), 1, 1),
                        stream=stream)
        return output

    def backward(self, grad_output):
        ##NOTE: IMPORTANT CONTIGUOUS
        # TODO: input
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = t.zeros(self.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                self.N, self.spatial_scale, C, H, W, self.outh, self.outw,
                grad_input.numel()]
        self.backward_fn(args=args,
                         block=(CUDA_NUM_THREADS, 1, 1),
                         grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                         stream=stream
                         )
        return grad_input, None
class RoIPooling2D(nn.Module):

    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)

class DuplicationRemovalNetwork(nn.Module):
    def __init__(self,n_relations = 16, appearance_feature_dim=1024,num_classes=20,d_f=128):
        super(DuplicationRemovalNetwork, self).__init__()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.key_feature_dim = int(appearance_feature_dim/n_relations)
        self.geo_feature_dim = int(appearance_feature_dim/n_relations)
        self.appearance_feature_dim=appearance_feature_dim
        self.n_class = num_classes+1

        self.nms_rank_fc = nn.Linear(appearance_feature_dim, d_f, bias=True)
        self.roi_feat_embedding_fc = nn.Linear(appearance_feature_dim,d_f,bias=True)
        self.relation_module = RelationModule(n_relations=n_relations,appearance_feature_dim=d_f,
                                              key_feature_dim=64,
                                              geo_feature_dim=64,isDuplication=True)

        self.nms_logit_fc = nn.Linear(appearance_feature_dim,1,bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,sample_roi,roi_cls_loc, roi_score, appearance_features,size):
        N = sample_roi.shape[0]
        roi_score = roi_score.data
        roi_cls_loc = roi_cls_loc.data
        roi = at.totensor(sample_roi)


        mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
        std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = at.totensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, self.n_class , 4)
        # clip bounding box
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

        prob = F.softmax(at.tovariable(roi_score), dim=1)

        prob,prob_argmax = torch.max(prob,dim=-1)
        cls_bbox = cls_bbox[np.arange(start=0,stop=N),prob_argmax]

        nonzero_idx=torch.nonzero(prob_argmax)

        if(nonzero_idx.size()[0]==0):
            return None,None,None
        else:
            nonzero_idx = nonzero_idx[:, 0]
            prob_argmax = prob_argmax[nonzero_idx]
            prob = prob[nonzero_idx]
            cls_bbox = cls_bbox[nonzero_idx]
            appearance_features_nobg = appearance_features[nonzero_idx]
            sorted_score,prob_argsort = torch.sort(prob,descending=True)

            sorted_prob = prob[prob_argsort]
            sorted_cls_bboxes = cls_bbox[prob_argsort]
            sorted_labels =  prob_argmax[prob_argsort]
            sorted_features = appearance_features_nobg[prob_argsort]

            nms_rank_embedding = RankEmbedding(sorted_prob.size()[0],self.appearance_feature_dim)
            nms_rank = self.nms_rank_fc(nms_rank_embedding)
            roi_feat_embedding = self.roi_feat_embedding_fc(sorted_features)
            nms_embedding_feat = nms_rank + roi_feat_embedding
            position_embedding = PositionalEmbedding(sorted_cls_bboxes,dim_g = self.geo_feature_dim)
            nms_logit = self.relation_module([sorted_features, nms_embedding_feat,position_embedding])
            nms_logit = self.nms_logit_fc(nms_logit)
            s1 = self.sigmoid(nms_logit).view(-1)
            nms_scores = s1 * sorted_prob

            return nms_scores, sorted_labels-1, sorted_cls_bboxes
class RelationModule(nn.Module):
    def __init__(self,n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64, isDuplication = False):
        super(RelationModule, self).__init__()
        self.isDuplication=isDuplication
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))
    def forward(self, input_data ):
        if(self.isDuplication):
            f_a, embedding_f_a, position_embedding =input_data
        else:
            f_a, position_embedding = input_data
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                if(self.isDuplication):
                    concat = self.relation[N](embedding_f_a,position_embedding)
                else:
                    concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                if(self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat+f_a
class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f_a, position_embedding):
        N,_ = f_a.size()

        position_embedding = position_embedding.view(-1,self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N,1,self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 )
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N,N)
        w_a = scaled_dot.view(N,N)

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N,N,1)
        w_v = w_v.view(N,1,-1)

        output = w_mn*w_v

        output = torch.sum(output,-2)
        return output

class RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 in_channels = 128,fc_features = 1024, n_relations = 0 , classifier = None):
        # n_class includes the background
        super(RoIHead, self).__init__()
        if classifier is None:
            self.n_relations=n_relations
            fully_connected1 = nn.Linear(7*7*in_channels, fc_features)
            relu1 = nn.ReLU(inplace=True)

            fully_connected2 = nn.Linear(fc_features, fc_features)
            relu2 = nn.ReLU(inplace=True)
            if(n_relations>0):
                self.dim_g = int(fc_features/n_relations)
                relation1= RelationModule(n_relations = n_relations, appearance_feature_dim=fc_features,
                                           key_feature_dim = self.dim_g, geo_feature_dim = self.dim_g)

                relation2 = RelationModule(n_relations=n_relations, appearance_feature_dim=fc_features,
                                            key_feature_dim=self.dim_g, geo_feature_dim=self.dim_g)
                self.classifier = nn.Sequential(fully_connected1, relu1, relation1,
                                                fully_connected2, relu2, relation2)
            else:
                self.classifier = nn.Sequential(fully_connected1, relu1,
                                                fully_connected2, relu2)
        else :
            self.classifier = classifier

        self.cls_loc = nn.Linear(fc_features, n_class * 4)
        self.score = nn.Linear(fc_features, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())
        if(self.n_relations>0):
            position_embedding = PositionalEmbedding(indices_and_rois[:, 1:],dim_g = self.dim_g)

        pool = self.roi(x, indices_and_rois)

        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores, fc7

class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16
    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores, fc7

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()