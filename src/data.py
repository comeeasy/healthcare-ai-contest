import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 

from lightning import LightningDataModule

from torchvision.datasets import MNIST, CIFAR10

from cfg import CFG
from src.utils import (
    open_json, 
    seperate_teeth_to_tooth_info,
    tooth_num_to_index,
    tooth_position_to_index,
    tooth_num_to_one_hot_vec,
    tooth_position_to_one_hot_vec
)
from glob import glob



class TeethDatasetSample(Dataset):
    def __init__(self, json_files, transform=None):
        self.json_files = json_files
        self.num_classes_of_tooth_position = 5 # front, upper, left, lower, right
        # https://info.singident.com/1
        self.num_classes_of_tooth_num = 32 # 11,.., 18, 21,..,28, 31,..38, 41,..,48 
        
        self.transform = transform
        
    def __len__(self):
        # A train batch consists of image of every each class 
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        
        teeth_dict_list = seperate_teeth_to_tooth_info(json_path, CFG.train_dataset_dir)

        # 구강 이미지에서 이빨 하나만 샘플 
        tooth = np.random.choice(teeth_dict_list, CFG.batch_size)

        # get data        
        tooth_img = tooth['teeth_image']
        tooth_num = tooth['teeth_num']
        tooth_position = tooth['teeth_position']
        target = tooth['target']
        
        # target = 1 if target =='true' else 0
        target = 1 if target else 0

        # image transform
        if self.transform:
            tooth_img = self.transform(tooth_img)
        
        # tooth_num, tooth_position -> one-hot vec 
        tooth_num_idx = tooth_num_to_index(tooth_num)
        tooth_num_one_hot = F.one_hot(tooth_num_idx, num_classes=self.num_classes_of_tooth_num) # [1, 32]
        tooth_num_one_hot = tooth_num_one_hot.squeeze() # [32]
        tooth_position_idx = tooth_position_to_index(tooth_position)
        tooth_position_one_hot = F.one_hot(tooth_position_idx, num_classes=self.num_classes_of_tooth_position) # [1, 5]
        tooth_position_one_hot = tooth_position_one_hot.squeeze() # [5]
        
        # [B x 1 x ...] 이 되도록 가장 앞선 shape을 맞춤
        tooth_img = tooth_img.unsqueeze(0)
        tooth_num_one_hot = tooth_num_one_hot.unsqueeze(0)
        tooth_position_one_hot = tooth_position_one_hot.unsqueeze(0)
        target = target.unsqueeze(0)
        
        return (tooth_img, tooth_num_one_hot, tooth_position_one_hot), target
    
class TeethDataset(Dataset):
    def __init__(self, json_files, transform=None):
        self.json_files = json_files
        self.num_classes_of_tooth_position = 5 # front, upper, left, lower, right
        # https://info.singident.com/1
        self.num_classes_of_tooth_num = 32 # 11,.., 18, 21,..,28, 31,..38, 41,..,48 
        
        self.transform = transform
        
    def __len__(self):
        # A train batch consists of image of every each class 
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        
        teeth_dict_list = seperate_teeth_to_tooth_info(json_path, CFG.train_dataset_dir)

        # find decayed tooth
        indices_decayed = []
        for i, tooth_info in enumerate(teeth_dict_list):
            if tooth_info["target"]:
                indices_decayed.append(i)
        
        # choice at least one decayed tooth
        if indices_decayed:
            decayed_tooth_info_idx = np.random.choice(indices_decayed)
            decayed_tooth_info = teeth_dict_list[decayed_tooth_info_idx]
            random_tooth_info = np.random.choice(teeth_dict_list)
            two_tooth_infos = [decayed_tooth_info, random_tooth_info]
        # if there is no decayed tooth, then just sample two tooths
        else:
            two_tooth_infos = np.random.choice(teeth_dict_list, 2)

        tooth_imgs_batch = torch.stack([CFG.val_transforms(tooth_info['teeth_image']) for tooth_info in two_tooth_infos])
        tooth_tooth_num_batch = torch.stack([tooth_num_to_one_hot_vec(tooth_info["teeth_num"]) for tooth_info in two_tooth_infos])
        tooth_tooth_position_batch = torch.stack([tooth_position_to_one_hot_vec(tooth_info["teeth_position"]) for tooth_info in two_tooth_infos])
        target_batch = torch.stack([torch.LongTensor([1]) if tooth_info["target"] else torch.LongTensor([0]) for tooth_info in two_tooth_infos])
        
        return (tooth_imgs_batch, tooth_tooth_num_batch, tooth_tooth_position_batch), target_batch

class TeethDatasetTest(Dataset):
    def __init__(self, json_files, transform=None):
        self.json_files = json_files
        self.transform = transform
        
    def __len__(self):
        # A train batch consists of image of every each class 
        return len(self.json_files)
    
    def __getitem__(self, idx):
        test_json_file = self.json_files[idx]
        teeth_dict_list = seperate_teeth_to_tooth_info(test_json_file, CFG.test_dataset_dir)
        
        tooth_imgs_batch = torch.stack([CFG.val_transforms(tooth_info['teeth_image']) for tooth_info in teeth_dict_list])
        tooth_tooth_num_batch = torch.stack([tooth_num_to_one_hot_vec(tooth_info["teeth_num"]) for tooth_info in teeth_dict_list])
        tooth_tooth_position_batch = torch.stack([tooth_position_to_one_hot_vec(tooth_info["teeth_position"]) for tooth_info in teeth_dict_list])
        target_batch = torch.stack([torch.LongTensor([1]) if tooth_info["target"] else torch.LongTensor([0]) for tooth_info in teeth_dict_list])
        
        return (tooth_imgs_batch, tooth_tooth_num_batch, tooth_tooth_position_batch), target_batch

class TeethDataModule(LightningDataModule):
    def __init__(
        self, 
    ) -> None:
        super().__init__()
        self.batch_size=CFG.batch_size
        self.width = CFG.img_transform_size_W
        self.height = CFG.img_transform_size_H
        self.test_size = CFG.test_size
    
        self.path = CFG.train_dataset_dir
        self.json_files = glob(os.path.join(self.path, "json", "*"))
        self.test_json_files = glob(os.path.join(CFG.test_dataset_dir, "json", "*"))
    
        self.train_transforms = CFG.train_transforms
        self.val_transforms = CFG.val_transforms
    
    def setup(self, stage: str="None") -> None:
        train_json_files, val_json_files = train_test_split(self.json_files, test_size=self.test_size)
        
        self.train_dateset = TeethDataset(train_json_files, transform=self.train_transforms)
        self.val_dataset = TeethDataset(val_json_files, transform=self.val_transforms)
        self.test_dataset = TeethDatasetTest(self.test_json_files, transform=self.val_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.train_dateset, shuffle=True, batch_size=self.batch_size,
            num_workers=8, pin_memory=True, drop_last=True)
    def val_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.val_dataset, batch_size=self.batch_size, 
            num_workers=8, pin_memory=True)
    def test_dataloader(self) -> DataLoader:
        return DataLoader( # type: ignore
            self.test_dataset, batch_size=1, 
            num_workers=8, pin_memory=True)