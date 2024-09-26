from torch.utils.data import Dataset
import os 
from glob import glob
from torchvision import transforms
import pydicom
import torch

class WbDataset(Dataset):
    def __init__(self, rootPath, dose, pet2suv):
        super().__init__()
        self.d2s = pet2suv
        lowDoseDict = {
            'D4': '2.886 x 600 WB D4',
            'D10': '2.886 x 600 WB D10',
            'D20': '2.886 x 600 WB D20',
            'D50': '2.886 x 600 WB D50',
            'D100': '2.886 x 600 WB D100',
            'Normal': '2.886 x 600 WB NORMAL',
        }
        self.patientList = glob(f"{rootPath}/*")
        self.lqPatientList = [ os.path.join(patient, lowDoseDict[dose])
                            for patient in self.patientList]
        self.gtList = [os.path.join(patient, lowDoseDict['Normal'])
                            for patient in self.patientList]
        
        self.perProc = transforms.Compose([
            transforms.Lambda(lambda img: self.d2s.get_SUV(pydicom.dcmread(img))),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.unsqueeze(-1)),
        ])

    def __getitem__(self, index):
        lqPatient = glob(f"{self.lqPatientList[index]}/*")
        lqPatientSUV = torch.cat([self.perProc(patient) for patient in lqPatient], dim=-1)
        hqPatient = glob(f"{self.gtList[index]}/*")
        hqPatinetSUV = torch.cat([self.perProc(patient) for patient in hqPatient], dim=-1)

        return {
            'lq' : lqPatientSUV, 'hq' : hqPatinetSUV
        }

    def __len__(self):
        return len(self.lqPatinetList)
            