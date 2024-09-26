from torch.utils.data import Dataset
from glob import glob
import os
from random import randint
from torchvision import transforms
import pydicom
import torch

class CentorBlockDataset(Dataset):
    def __init__(self, rootPath, dose, blockShape, numBlock, pet2suv):
        super().__init__()
        self.H, self.W, self.D = blockShape
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
        self.centorBlockList = [item for item in self.lqPatientList for _ in range(numBlock)]

        self.gtList = [os.path.join(patient, lowDoseDict['Normal'])
                            for patient in self.patientList]
        self.gtBlockList = [item for item in self.gtList for _ in range(numBlock)]

        self.perProc = transforms.Compose([
            transforms.Lambda(lambda img: self.d2s.get_SUV(pydicom.dcmread(img, force=True))),
            transforms.ToTensor(),
            transforms.CenterCrop((self.H, self.W)),
        ])
        
    def __getitem__(self, index):
        patient = glob(f"{self.centorBlockList[index]}/*")
        numLayer = len(patient)
        centorCursor = randint(0, numLayer - self.D - 1)
        patientSubset = [patient[i] for i in range(centorCursor, centorCursor + self.D)]
        patientBlock = [self.perProc(patinet) for patinet in patientSubset]
        lqBlock = torch.stack(patientBlock, dim=-1).to(torch.float32)
        lqBlock = lqBlock.permute(0, 3, 1, 2)

        gt = glob(f"{self.gtBlockList[index]}/*")
        gtSubset = [gt[i] for i in range(centorCursor, centorCursor + self.D)]
        gtBlock = [self.perProc(gt) for gt in gtSubset]
        hqBlock = torch.stack(gtBlock, dim=-1).to(torch.float32)
        hqBlock = hqBlock.permute(0, 3, 1, 2)

        return {
            "lq": lqBlock, "hq": hqBlock
        }
    
    def __len__(self):
        return len(self.centorBlockList)

