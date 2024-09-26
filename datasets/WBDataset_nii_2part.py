from torch.utils.data import Dataset
import os 
from glob import glob
from torchvision import transforms
import pydicom
import torch
import pandas as pd
import nibabel as nib

class WbDataset_nii_2parts(Dataset):
    def __init__(self, rootPath, dose, csvName, pet2suv, shape):
        super().__init__()
        self.d2s = pet2suv
        lowDoseDict = {
            "D100" : '100',
            "D50" : '50', 
            "D20" : '20',
            "D10": '10',
            "D4" : '4',
        }
        self.patientInfo = pd.read_csv(csvName)
        self.patientSUVInfoList = []
        for index, row in self.patientInfo.iterrows():
            if row["DRF"] == lowDoseDict[dose]:
                self.patientSUVInfoList.append({
                    "niiPath" : glob(f"{os.path.join(rootPath, row['PID'])}/*")[0],
                    "dose" : row["Dose"], 
                    "weight": row["Weight"],
                })
        
        self.perProc = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(shape[0])
        ])

    def __getitem__(self, index):
        patient = self.patientSUVInfoList[index]
        nii_data, nii_affine = self.d2s.get_SUV(patient["niiPath"], patient["weight"], patient["dose"])
        nii_data = self.perProc(nii_data)
        nii_data_Upper = nii_data[:640, :, :]
        
        nii_data_Upper = nii_data_Upper.permute(0, 1, 2).unsqueeze(0).to(torch.float32)
        
        return {
            "upper": nii_data_Upper,
            "dose": patient["dose"],
            "weight": patient["weight"],
            "affine": nii_affine,
            "dir": patient["niiPath"].split("/")[-2], 
            "name":  patient["niiPath"].split("/")[-1],
        }

    def __len__(self):
        return len(self.patientSUVInfoList)


            
