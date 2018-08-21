from lib.eval_tool import eval_detection_voc
from torch.utils import data as data_
import model
import time
import torch

from data.dataset import TestDataset
from config import opt


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


testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    shuffle=False, \
                                    # pin_memory=True
                                    )
resnet = model.resnet18(20,True)
resnet = resnet.cuda()
resnet = torch.nn.DataParallel(resnet).cuda()
resnet.load_state_dict(torch.load('Weights/resnet_15.pt'))

resnet.eval()

eval_result = eval(test_dataloader, resnet, test_num=10000)
print('eval_result : ',eval_result)