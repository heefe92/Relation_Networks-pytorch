import collections

import numpy as np
from torch.utils import data as data_
import model

import torch
import torch.optim as optim
from data.dataset import inverse_normalize
from lib.eval_tool import eval_detection_voc
from data.dataset import Dataset, TestDataset
from config import opt
import cv2,time


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


def run_train():
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset, \
                                      batch_size=opt.batch_size, \
                                      shuffle=True, \
                                      # pin_memory=True,
                                      num_workers=opt.num_workers)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=False#, \
                                       #pin_memory=True
                                       )

    resnet = model.resnet18(20,True)
    resnet = resnet.cuda()
    resnet = torch.nn.DataParallel(resnet).cuda()

    #resnet.load_state_dict(torch.load('Weights/resnet18_12.pt'))

    ## resnet params freeze
    child_count=0
    for child in resnet.module.children():
        if(child_count<8):
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
        child_count+=1


    optimizer = optim.Adam(resnet.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    resnet.train()
    resnet.module.use_preset(isTraining=True)
    resnet.module.freeze_bn()

    epoch_loss_hist = []
    for epoch_num in range(opt.epoch):
        start= time.time()
        resnet.train()
        resnet.module.use_preset(isTraining=True)
        resnet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader):
            optimizer.zero_grad()
            losses = resnet([data[0].cuda().float(),data[1].cuda().float(),data[2].cuda().float(),data[3].cuda().float()])
            losses[4].backward()
            torch.nn.utils.clip_grad_norm_(resnet.parameters(), 0.1)

            optimizer.step()

            curr_loss=losses[4].item()
            loss_hist.append(float(curr_loss))

            epoch_loss.append(float(curr_loss))

            if(iter_num % 12000==11999):
                print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(curr_loss), np.mean(loss_hist)))

            del curr_loss
        print('1epoch time :',time.time()-start)
        print('Epoch: {} | epoch loss: {:1.5f}'.format(
            epoch_num, np.mean(epoch_loss)))
        scheduler.step(np.mean(epoch_loss))
        epoch_loss_hist.append(np.mean(epoch_loss))
        if(epoch_num%1 == 0):
            resnet.eval()
            if(epoch_num<15):
                eval_result = eval(test_dataloader, resnet, test_num=1000)
            else :
                eval_result = eval(test_dataloader, resnet, test_num=10000)
            print(epoch_num,'_eval_result : ', eval_result)
            torch.save(resnet.state_dict(), 'Weights/resnet18_{}.pt'.format(epoch_num))

    print(epoch_loss_hist)

if __name__=="__main__":
    run_train()