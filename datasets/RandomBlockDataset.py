from torch.utils.data import Dataset
from glob import glob
import os
from random import randint
from torchvision import transforms
import pydicom
import torch

class RandomBlockDataset(Dataset):
    def __init__(self, rootPath, dose, pet2suv, numBlock=20, blockShape=(122, 122, 122), volShape=(673, 360, 360)):
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
        self.lqBlockList = [item for item in self.lqPatientList for _ in range(numBlock)]

        self.gtList = [os.path.join(patient, lowDoseDict['Normal'])
                            for patient in self.patientList]
        self.gtBlockList = [item for item in self.gtList for _ in range(numBlock)]

        self.perProc = transforms.Compose([
            transforms.Lambda(lambda img: self.d2s.get_SUV(pydicom.dcmread(img))),
            transforms.ToTensor(),
        ])
        self.bH = blockShape[1]; self.bW = blockShape[2]; self.bD = blockShape[0]
        self.vH = volShape[1]; self.vW = volShape[2]; self.vD = volShape[0]

    def __getitem__(self, index):
        centor_z = randint(100 , self.vD - self.bD - 200)
        centor_x = randint(52, self.vH - self.bH - 52)
        centor_y = randint(52, self.vW - self.bW - 52)

        patients = glob(f"{self.lqBlockList[index]}/*")
        lq_block = torch.cat([self.perProc(patinet).to(torch.float32) for patinet in patients], dim=0)
        lq_img = lq_block[centor_z : centor_z + self.bD, centor_x : centor_x + self.bH, centor_y : centor_y + self.bW].unsqueeze(0)

        gts = glob(f"{self.gtBlockList[index]}/*")
        gt_block = torch.cat([self.perProc(gt).to(torch.float32) for gt in gts], dim=0)
        gt_img = gt_block[centor_z : centor_z + self.bD, centor_x : centor_x + self.bH, centor_y : centor_y + self.bW].unsqueeze(0)

        return {
            "lq": lq_img, "hq": gt_img
        }
    
    def __len__(self):
        return len(self.lqBlockList)

