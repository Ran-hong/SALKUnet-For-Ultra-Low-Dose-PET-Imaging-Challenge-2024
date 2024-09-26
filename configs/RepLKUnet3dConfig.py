from configs.defaultConfig import getDefalutConfigs
import ml_collections

def getConfigs():
    config = getDefalutConfigs()
    config.mode.mode = "train"

    config.data.name = "RandomBlockDataset" 
    config.data.RandomBlock = ml_collections.ConfigDict()
    config.data.RandomBlock.dose = "D100"
    config.data.RandomBlock.numBlock = 20
    config.data.RandomBlock.blockShape = (112, 112, 112)
    config.data.RandomBlock.volShape = (673, 360, 360)
    config.data.trainRootPath = "/home/uPET/dataset/training/PART2"
    config.data.evalRootPath = "/home/uPET/dataset/eval/PART2"

    config.dicom2SUV.name = "sample_way"

    config.runner.type = "Runner_e2e"

    # train
    config.train.num_train_epoch = 50
    config.train.batch_size = 4
    config.train.snapshot_freq = 1
    config.train.snapshot_freq_for_preemption = 1

    # model
    config.model.name = "RepLKUnet3d"
    config.model.RepLKUnet = ml_collections.ConfigDict()
    config.model.RepLKUnet.in_channels = 1
    config.model.RepLKUnet.out_channels = 1
    config.model.RepLKUnet.mlp_ratio = 4
    config.model.RepLKUnet.features = [16, 32, 64, 128]
    config.model.RepLKUnet.lkSize = [17, 17, 17, 13]
    config.model.RepLKUnet.skSize = [3, 3, 3, 3]

    # optim
    config.optim.lr = 2e-4
    config.optim.betas = (0.9, 0.999)
    config.optim.eps = 1e-8
    config.optim.weight_decay = 0

    return config