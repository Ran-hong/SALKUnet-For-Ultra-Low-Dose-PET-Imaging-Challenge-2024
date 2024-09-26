from configs.defaultConfig import getDefalutConfigs
import ml_collections

def getConfigs():
    config = getDefalutConfigs()
    config.mode.mode = "train"

    config.data.name = "CentorBlock" 
    config.data.CentorBlock = ml_collections.ConfigDict()
    config.data.trainRootPath = "/home/uPET/dataset/training/PART2"
    config.data.evalRootPath = "/home/uPET/dataset/eval/PART2"
    config.data.CentorBlock.dose = "D10"
    config.data.CentorBlock.numBlock = 10
    config.data.CentorBlock.blockShape = (224, 224, 224)

    config.dicom2SUV.name = "sample_way"

    config.runner.type = "Runner_e2e"

    # train
    config.train.num_train_epoch = 25
    config.train.batch_size = 2
    config.train.snapshot_freq = 1
    config.train.snapshot_freq_for_preemption = 1

    # model
    config.model.name = "Unet3d_AKAB"
    config.model.Unet3d = ml_collections.ConfigDict()
    config.model.Unet3d.in_channels = 1
    config.model.Unet3d.out_channels = 1
    config.model.Unet3d.init_features = 16

    # optim
    config.optim.lr = 2e-4
    config.optim.betas = (0.9, 0.999)
    config.optim.eps = 1e-8
    config.optim.weight_decay = 0

    return config