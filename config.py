
class Config:
    # data
    voc_data_dir = '/media/heecheol/새 볼륨/DataSet/VOC2007/'
    min_size = 604  # image resize
    max_size = 1024 # image resize
    num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    lr = 1e-4


    # training
    epoch = 100
    isLearnNMS = False
    #The batch size can still only one.
    batch_size=1

    model_name='resnet101_relation'

    features_dim = 512



opt = Config()
