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
    config.data.WbDatasetNii2part.shape = (256, 256)
    

    config.dicom2SUV.name = "sample_way_4nii"

    config.runner.type = "Runner_e2e"


    # model
    config.model.name = "Unet3d_deep"
    config.model.Unet3d_deep = ml_collections.ConfigDict()
    config.model.Unet3d_deep.in_channels = 1
    config.model.Unet3d_deep.out_channels = 1
    config.model.Unet3d_deep.init_features = 16

    # optim
    config.optim.lr = 1e-3
    config.optim.betas = (0.9, 0.999)
    config.optim.eps = 1e-8
    config.optim.weight_decay = 0

    return config