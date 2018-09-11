from torch.utils import data as data_
import model

import torch
from lib.eval_tool import eval_detection_voc
from data.dataset import TestDataset
from config import opt
import cv2,time
import numpy as np
from lib.array_tool import tonumpy
def eval(dataloader, resnet, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, data in enumerate(dataloader):
        (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) = data
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = resnet.module.predict(imgs, [sizes])
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
    resnet = resnet.cuda()
    resnet = torch.nn.DataParallel(resnet).cuda()

    resnet.load_state_dict(torch.load('Weights/resnet101_pyramid_no_relation_e2e_14.pt'))
    resnet.module.use_preset(isTraining=False,preset='evaluate')
    for ii, data in enumerate(test_dataloader):
        print(data[2])
        print(data[3])
        nms_scores, sorted_labels, sorted_cls_bboxes = resnet(
            data[0].cuda().float())
        if not ( nms_scores is None):
            test = np.reshape(np.argwhere(nms_scores>0.2),-1)
            nms_scores = nms_scores[test]
            sorted_labels = sorted_labels[test]
            sorted_cls_bboxes = sorted_cls_bboxes[test]

            print(nms_scores)
            print(sorted_labels)
            print(sorted_cls_bboxes)
    #print(eval(test_dataloader,resnet))

if __name__ == "__main__":
    run_evaluate()