from configs.defaultConfig import getDefalutConfigs
import ml_collections

def getConfigs():
    config = getDefalutConfigs()
    config.mode.mode = "test"

    config.data.name = "WbDataset_nii" 
    config.data.WbDatasetNii = ml_collections.ConfigDict()
    config.data.trainRootPath = "/home/uPET/dataset/training/PART2"
    config.data.evalRootPath = "/home/uPET/dataset/eval/PART2"
    config.data.testRootPath = "/home/uPET/dataset/testing/Second_round_test_dataset/test"
    config.data.WbDatasetNii.dose = "D100"
    config.data.WbDatasetNii.csvName = "/home/uPET/dataset/testing/Second_round_test_dataset/meta_info.csv"
    config.data.WbDatasetNii.shape = (224, 224)
    

    config.dicom2SUV.name = "sample_way_4nii"

    config.runner.type = "Runner_e2e"


    # model
    config.model.name = "Unet3d"
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