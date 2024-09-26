from dicom2suv.Dicom2SUV import Dicom2SUV
import nibabel as nib
import numpy as np

class Sample_way(Dicom2SUV):
    def __init__(self):
        pass
    def get_SUV(self, ds):
        image_data = ds.pixel_array
        
        # 提取患者体重 (kg)
        patient_weight = ds.PatientWeight  # 例如: 70.0 kg
        
        # 提取注射的放射性剂量 (MBq)
        radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = radiopharmaceutical_info.RadionuclideTotalDose  # 例如: 50000000 Bq
        
        # 提取图像的放射性活度 (Bq/mL)
        # 这里假设 DICOM 文件中存储的是活度浓度
        activity_concentration = image_data  # 例如: 3.5 Bq/mL
        
        # 计算 SUV
        suv = (activity_concentration * patient_weight * 1000) / injected_dose
        
        return suv
    
    def get_dicom(self, suv, ds):
        image_data = ds.pixel_array
        
        # 提取患者体重 (kg)
        patient_weight = ds.PatientWeight  # 例如: 70.0 kg
        
        # 提取注射的放射性剂量 (MBq)
        radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = radiopharmaceutical_info.RadionuclideTotalDose  # 例如: 50000000 Bq
        
        # 提取图像的放射性活度 (Bq/mL)
        # 这里假设 DICOM 文件中存储的是活度浓度
        activity_concentration = (suv * injected_dose) / (patient_weight * 1000)

        return activity_concentration

class Sample_way_4nii(Dicom2SUV):
    def __init__(self):
        super().__init__()
    def get_SUV(self, niiPath, weight, dose):
        image_data = nib.load(niiPath).get_fdata()
        image_affine = nib.load(niiPath).affine
        image_data_list = []
        for i in range(image_data.shape[-1]):
            image_layer = image_data[:, :, i]
            suv_layer = (image_layer * weight * 1000) / (dose) 
            image_data_list.append(suv_layer)
        image_data = np.stack(image_data_list, axis=-1)
        

        return image_data, image_affine
    
    def get_dicom(self, suv, weight, dose):
        activity_concentration = (suv * dose) / (weight * 1000)
        return activity_concentration
        


