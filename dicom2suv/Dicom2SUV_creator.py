from dicom2suv.sample_way import Sample_way
from dicom2suv.sample_way import Sample_way_4nii

def Dicom2SUV_factory(config):
    d2s_config = config.dicom2SUV
    if d2s_config.name == "sample_way":
        d2s = Sample_way()
    elif d2s_config.name == "sample_way_4nii":
        d2s = Sample_way_4nii()
    else:
        raise f"The way of dicom to SUV {d2s_config.name} not exist!"
    return d2s

