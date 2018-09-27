from torch.utils import data as data_
import model

import torch
from lib.eval_tool import eval_detection_voc
from data.dataset import TestDataset
from config import opt
import cv2,time
import numpy as np
from lib.array_tool import tonumpy

# def eval(dataloader, model, test_num=10000):
#     pred_bboxes, pred_labels, pred_scores = list(), list(), list()
#     gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
#     for ii, data in enumerate(dataloader):
#         (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) = data
#
#         nms_scores, sorted_labels, sorted_cls_bboxes = model.predict(
#             imgs.cuda().float())
#         if not ( nms_scores is None):
#             test = np.reshape(np.argwhere(nms_scores>0.7),-1)
#             nms_scores = nms_scores[test]
#             sorted_labels = sorted_labels[test]
#             sorted_cls_bboxes = sorted_cls_bboxes[test]
#
#             pred_bboxes.append(np.reshape(tonumpy(sorted_cls_bboxes),(-1,4)).copy())
#             pred_labels.append(np.reshape(tonumpy(sorted_labels),(-1)).copy())
#             pred_scores.append(np.reshape(tonumpy(nms_scores),(-1)).copy())
#         else:
#             pred_bboxes.append(np.array([]))
#             pred_labels.append(np.array([]))
#             pred_scores.append(np.array([]))
#         gt_bboxes += list(gt_bboxes_.numpy())
#         gt_labels += list(gt_labels_.numpy())
#         gt_difficults += list(gt_difficults_.numpy())
#         if ii == test_num: break
#     result = eval_detection_voc(
#         pred_bboxes, pred_labels, pred_scores,
#         gt_bboxes, gt_labels, gt_difficults,
#         use_07_metric=True)
#     return result


def eval(dataloader, model, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, data in enumerate(dataloader):
        (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) = data
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def run_evaluate():
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                        batch_size=opt.batch_size,
                                        num_workers=opt.num_workers,
                                        shuffle=False#, \
                                        #pin_memory=True
                                        )

    resnet = model.resnet101(20,True)
    resnet = torch.nn.DataParallel(resnet).cuda()

    resnet.load_state_dict(torch.load('Weights/resnet101_relation_47.pt'))
    resnet.module.use_preset(isTraining=False,preset='evaluate')
    resnet.eval()

    for child in resnet.module.children():
        for param in child.parameters():
            param.requires_grad = False

    print(eval(test_dataloader,resnet,10000))

if __name__ == "__main__":
    run_evaluate()