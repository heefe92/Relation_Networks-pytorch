
class Config:
    # data
    voc_data_dir = '/media/heecheol/새 볼륨/DataSet/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0001
    lr = 1e-4


    # training
    trainset = 'trainval'
    testset = 'test'
    epoch = 15
    isLearnNMS = False
    use_adam = True #You need set a very low lr for Adam
    #The batch size can still only one.
    batch_size=1

    model_name='squeeze'

    features_dim = 512



opt = Config()
