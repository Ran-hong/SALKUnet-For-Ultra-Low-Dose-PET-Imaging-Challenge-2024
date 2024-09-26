from datasets.CentorBlockDataset import CentorBlockDataset
from datasets.WBDataset_nii import WbDataset_nii
from datasets.RandomBlockDataset import RandomBlockDataset
from datasets.WBDataset_nii_2part import WbDataset_nii_2parts
from dicom2suv.Dicom2SUV_creator import Dicom2SUV_factory

def datasetFactory(config):
    d2s = Dicom2SUV_factory(config)
    if config.mode.mode == "train":
        rootPath = config.data.trainRootPath
    elif config.mode.mode == "eval":
        rootPath = config.data.evalRootPath
    elif config.mode.mode == "test":
        rootPath = config.data.testRootPath
    if config.data.name == "CentorBlock":
        data = config.data.CentorBlock
        return CentorBlockDataset(
            rootPath = rootPath,
            dose = data.dose,
            blockShape = data.blockShape,
            numBlock = data.numBlock,
            pet2suv = d2s,
        )
    elif config.data.name == "WbDataset_nii":
        data = config.data.WbDatasetNii
        return WbDataset_nii(
            rootPath = rootPath,
            dose = data.dose,
            csvName= data.csvName,
            pet2suv=d2s,
            shape=data.shape,
        )
    elif config.data.name == "RandomBlockDataset":
        data = config.data.RandomBlock
        return RandomBlockDataset(
            rootPath = rootPath,
            dose = data.dose,
            pet2suv = d2s,
            numBlock = data.numBlock,
            blockShape = data.blockShape,
            volShape = data.volShape,
        )
    elif config.data.name == "WbDataset_nii_2parts":
        data = config.data.WbDatasetNii2part
        return WbDataset_nii_2parts(
            rootPath = rootPath,
            dose = data.dose,
            csvName= data.csvName,
            pet2suv=d2s,
            shape=data.shape,
        )
    else:
        raise f"The dataset name {config.data.name} not exist!"
