import ml_collections
import ml_collections.config_dict

def getDefalutConfigs():
    config = ml_collections.ConfigDict()
    config.mode = ml_collections.ConfigDict()
    config.data = ml_collections.ConfigDict()
    config.dicom2SUV = ml_collections.ConfigDict()
    config.runner = ml_collections.ConfigDict()
    config.train = ml_collections.ConfigDict()
    config.optim = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()

    return config