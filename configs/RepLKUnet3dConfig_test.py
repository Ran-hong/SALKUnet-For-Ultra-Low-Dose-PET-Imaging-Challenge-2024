from configs.defaultConfig import getDefalutConfigs
import ml_collections

def getConfigs():
    config = getDefalutConfigs()
    config.mode.mode = "test"

    config.data.name = "WbDataset_nii_2parts" 
    config.data.WbDatasetNii2part = ml_collections.ConfigDict()
    config.data.trainRootPath = "/home/uPET/dataset/training/PART2"
    config.data.evalRootPath = "/home/uPET/dataset/eval/PART2"
    config.data.testRootPath = "/home/uPET/dataset/testing/Second_round_test_dataset/test"
    config.data.WbDatasetNii2part.dose = "D100"
    config.data.WbDatasetNii2part.csvName = "/home/uPET/dataset/testing/Second_round_test_dataset/meta_info.csv"
    config.data.WbDatasetNii2part.shape = (224, 224)
    

    config.dicom2SUV.name = "sample_way_4nii"

    config.runner.type = "Runner_e2e"


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