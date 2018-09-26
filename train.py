import collections

import numpy as np
from torch.utils import data as data_
import model

from trainer import Trainer
import torch
import torch.optim as optim
from data.dataset import Dataset, TestDataset
from config import opt
import cv2,time

def run_train(train_verbose=False):
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

    resnet = model.resnet101(20,True).cuda()

    optimizer = optim.Adam(resnet.parameters(), lr=opt.lr)

    loss_hist = collections.deque(maxlen=500)
    epoch_loss_hist = []
    resnet_trainer = Trainer(resnet,optimizer,model_name=opt.model_name)

    freeze_num = 8 #pretrain model
    best_loss = 10
    best_loss_epoch_num = -1
    num_bad_epochs = 0
    max_bad_epochs = 5
    resnet_trainer.model_freeze(freeze_num=freeze_num)

    for epoch_num in range(opt.epoch):
        resnet_trainer.train_mode(freeze_num)
        train_start_time = time.time()
        train_epoch_loss = []
        start = time.time()
        for iter_num, data in enumerate(dataloader):
            curr_loss = resnet_trainer.train_step(data)
            loss_hist.append(float(curr_loss))
            train_epoch_loss.append(float(curr_loss))

            if (train_verbose):
                print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f} | Iter time: {:1.5f} | Train'
                      ' time: {:1.5f}'.format(epoch_num, iter_num, float(curr_loss), np.mean(loss_hist),
                       time.time()-start, time.time()-train_start_time))
                start = time.time()

            del curr_loss
        print('train epoch time :', time.time() - train_start_time)
        print('Epoch: {} | epoch train loss: {:1.5f}'.format(
            epoch_num, np.mean(train_epoch_loss)))

        vali_start_time = time.time()

        #resnet_trainer.eval_mode()
        vali_epoch_loss = []
        for iter_num, data in enumerate(test_dataloader):
            curr_loss = resnet_trainer.get_loss(data)
            vali_epoch_loss.append(float(curr_loss))

            del curr_loss

        epoch_loss_hist.append(np.mean(vali_epoch_loss))

        # vali_eval_result = resnet_trainer.run_eval(test_dataloader)
        # print(vali_eval_result)
        print('vali epoch time :', time.time() - vali_start_time)
        print('Epoch: {} | epoch vali loss: {:1.5f}'.format(
            epoch_num, np.mean(vali_epoch_loss)))

        if (best_loss < np.mean(vali_epoch_loss)):
            num_bad_epochs += 1
        else:
            best_loss = np.mean(vali_epoch_loss)
            best_loss_epoch_num = epoch_num
            num_bad_epochs = 0
            resnet_trainer.model_save(epoch_num)
        if (num_bad_epochs > max_bad_epochs):
            freeze_num=11
            num_bad_epochs = 0
            resnet_trainer.model_load(best_loss_epoch_num)
            resnet_trainer.reduce_lr(factor=0.1, verbose=True)
            resnet_trainer.model_freeze(freeze_num=freeze_num)

        print('best epoch num', best_loss_epoch_num)
        print('----------------------------------------')

    print(epoch_loss_hist)


if __name__ == "__main__":
    run_train(train_verbose = True)