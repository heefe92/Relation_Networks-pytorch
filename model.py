import time
import math
import torch.utils.model_zoo as model_zoo
import six

from torch.nn import functional as F
from losses import FasterRCNNLoss
from lib.nms import non_maximum_suppression
from collections import namedtuple
from string import Template
import lib.array_tool as at
from config import opt
from data.dataset import preprocess
from lib.bbox_tools import loc2bbox

import torch as t
from torch.autograd import Function

from lib.roi_cupy import kernel_backward, kernel_forward
from lib.creator_tool import ProposalCreator, ProposalTargetCreator, AnchorTargetCreator

Stream = namedtuple('Stream', ['ptr'])


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

import torch
import torch.nn as nn
import numpy as np
import cupy as cp

@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K
class ResNet(nn.Module):
    feat_stride = 32  # downsample 32x for output of convolution resnet
    def __init__(self, num_classes, block, layers):
        self.training=False
        self.inplanes = 64
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.n_class = num_classes+1

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels,
                         self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels,
                         self.layer4[layers[3]-1].conv3.out_channels]

        #self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],feature_size = 512)
        self.rpn = RegionProposalNetwork(in_channels=512,mid_channels=512,feat_stride = self.feat_stride)
        self.roi_head = RoIHead(n_class = num_classes+1,roi_size=7,spatial_scale=(1. / self.feat_stride),
                                in_channels=512,fc_features = 1024, n_relations= 0)

        self.proposal_target_creator = ProposalTargetCreator()
        self.anchor_target_creator = AnchorTargetCreator()

        self.Loss = FasterRCNNLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01

        self.freeze_bn()
    def use_preset(self,isTraining,preset='visualize'):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.5
        self.training=isTraining
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, scale=1.):
        if self.training:
            img_batch, bboxes, labels, scale = inputs

        else:
            img_batch = inputs

        scale = 1.
        _, _, H, W = img_batch.shape
        img_size = (H, W)
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #features = self.fpn([x2, x3, x4])
        features = x4
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features,img_size,scale)

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
            roi_cls_loc, roi_score = self.roi_head(features, sample_roi, sample_roi_index)

            return self.Loss(gt_rpn_loc, gt_rpn_label, gt_roi_loc, gt_roi_label, roi_cls_loc, roi_score, rpn_locs, rpn_scores)
        else:
            roi_cls_loc, roi_score = self.roi_head(features, rois, roi_indices)
            return roi_cls_loc,roi_score, rois, roi_indices

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
    def predict(self,imgs,sizes=None,visualize=False):
        if visualize:
            self.use_preset(isTraining=False,preset='visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            self.use_preset(isTraining=False,preset='evaluate')
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            if visualize:
                roi = at.totensor(rois) / scale
            else:
                roi = at.totensor(rois) / scale.cuda().float()

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

        # self.use_preset('evaluate')
        # self.train()
        return bboxes, labels, scores
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
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.ReLU(inplace=True)
        self.P3_downsampled = nn.MaxPool2d(2, stride=2,padding=0)
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.ReLU(inplace=True)
        self.P4_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_4 = nn.ReLU(inplace=True)
        self.P4_downsampled = nn.MaxPool2d(2, stride=2,padding=0)


        # add P4 elementwise to C3
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.ReLU(inplace=True)
        self.P5_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_4 = nn.ReLU(inplace=True)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P3_x = self.P3_1(C3)
        P3_x = self.P3_2(P3_x)
        P3_downsampled_x = self.P3_downsampled(P3_x)


        P4_x = self.P4_1(C4)
        P4_x = self.P4_2(P4_x)
        P4_x = P3_downsampled_x + P4_x
        P4_x = self.P4_3(P4_x)
        P4_x = self.P4_4(P4_x)
        P4_downsampled_x = self.P4_downsampled(P4_x)

        P5_x = self.P5_1(C5)
        P5_x = self.P5_2(P5_x)
        P5_x = P4_downsampled_x + P5_x
        P5_x = self.P5_3(P5_x)
        P5_x = self.P5_4(P5_x)

        return P5_x
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
    pass
class RelationModule(nn.Module):
    def __init__(self,n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationModule, self).__init__()
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))
    def forward(self, f_a, position_embedding):
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                concat = torch.cat((concat,self.relation[N](f_a,position_embedding)),-1)
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
                 in_channels = 128,fc_features = 1024, n_relations = 16):
        # n_class includes the background
        super(RoIHead, self).__init__()
        self.n_relations=n_relations
        self.fully_connected1 = nn.Linear(7*7*in_channels, fc_features)
        self.relu1 = nn.ReLU(inplace=True)
        if(n_relations>0):
            self.dim_g = int(fc_features/n_relations)
            self.relation1= RelationModule(n_relations = n_relations, appearance_feature_dim=fc_features,
                                       key_feature_dim = self.dim_g, geo_feature_dim = self.dim_g)

        self.fully_connected2 = nn.Linear(fc_features, fc_features)
        self.relu2 = nn.ReLU(inplace=True)
        if(n_relations>0):
            self.relation2= RelationModule(n_relations = n_relations, appearance_feature_dim=fc_features,
                                       key_feature_dim = self.dim_g, geo_feature_dim = self.dim_g)

        self.cls_loc = nn.Linear(fc_features, n_class * 4)
        self.score = nn.Linear(fc_features, n_class)

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
            position_embedding = self.embedding(indices_and_rois,dim_g = self.dim_g)

        pool = self.roi(x, indices_and_rois)

        pool = pool.view(pool.size(0), -1)
        fc1 = self.fully_connected1(pool)
        fc1 = self.relu1(fc1)

        if(self.n_relations>0):
            fc1 = self.relation1(fc1,position_embedding)

        fc2 = self.fully_connected2(fc1)
        fc2 = self.relu2(fc2)
        if(self.n_relations>0):
            fc2 = self.relation2(fc2,position_embedding)

        roi_cls_locs = self.cls_loc(fc2)
        roi_scores = self.score(fc2)
        return roi_cls_locs, roi_scores
    def embedding(self,f_g, wave_len=1000, dim_g=64):
        f_g = f_g[:,1:]
        x_min,y_min,x_max,y_max = torch.chunk(f_g, 4, dim=1)

        cx = (x_min + x_max)* 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        delta_x = cx - cx.view(1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min = 1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min = 1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(1,-1))
        delta_h = torch.log(h / h.view(1, -1))
        size = delta_h.size()

        delta_x = delta_x.view(size[0],size[1],1)
        delta_y = delta_y.view(size[0],size[1],1)
        delta_w = delta_w.view(size[0],size[1],1)
        delta_h = delta_h.view(size[0],size[1],1)

        position_mat = torch.cat((delta_x,delta_y,delta_w,delta_h),-1)

        feat_range = torch.arange(dim_g/8).cuda()
        dim_mat = feat_range/ (dim_g/8)
        dim_mat = 1. / (torch.pow(wave_len,dim_mat))



        dim_mat = dim_mat.view(1,1,1,-1)
        position_mat = position_mat.view(size[0],size[1],4,-1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(size[0],size[1],-1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat,cos_mat),-1)

        return embedding
def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model

def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model

def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model