import collections

import numpy as np
from torch.utils import data as data_
import model

from trainer import Trainer
import torch
import torch.optim as optim
from data.dataset import VGGDataset, VGGTestDataset
from config import opt
import cv2,time

def run_train(train_verbose=False):
    dataset = VGGDataset(opt)
    dataloader = data_.DataLoader(dataset, \
                                      batch_size=opt.batch_size, \
                                      shuffle=True, \
                                      # pin_memory=True,
                                      num_workers=opt.num_workers)

    testset = VGGTestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )

    my_model = model.SqueezeFRCN(20).cuda()

    optimizer = my_model.get_optimizer()

    loss_hist = collections.deque(maxlen=500)
    epoch_loss_hist = []
    my_trainer = Trainer(my_model,optimizer,model_name=opt.model_name)
    #my_trainer.model_load(3)

    freeze_num = -1 #pretrain model
    best_map = 0
    best_map_epoch_num = -1

    for epoch_num in range(opt.epoch):
        my_trainer.train_mode(freeze_num)
        train_start_time = time.time()
        train_epoch_loss = []
        start = time.time()
        for iter_num, data in enumerate(dataloader):
            curr_loss = my_trainer.train_step(data)
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


        vali_eval_result = my_trainer.run_eval(test_dataloader)
        print(vali_eval_result)
        vali_map = vali_eval_result['map']
        print('vali epoch time :', time.time() - vali_start_time)


        if(best_map < vali_map):
            best_map = vali_map
            best_map_epoch_num = epoch_num
            my_trainer.model_save(epoch_num)
        if (epoch_num==9):
            my_trainer.model_load(best_map_epoch_num)
            my_trainer.reduce_lr(factor=0.1, verbose=True)

        print('best epoch num', best_map_epoch_num)
        print('----------------------------------------')

    print(epoch_loss_hist)


if __name__ == "__main__":
    run_train(train_verbose = True)
    #my_model = model.SqueezeFRCN(20)