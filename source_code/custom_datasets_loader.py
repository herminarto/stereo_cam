# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:26:02 2022

@author: elektro
"""

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the left image paths
        self.image_left = np.asarray(self.data_info.iloc[:, 1])
        # Second column contains the right image paths
        self.image_right = np.asarray(self.data_info.iloc[:, 2])
        # Third column contains the disparity image paths
        self.image_disparity = np.asarray(self.data_info.iloc[:, 3])
        # Fourth column contains the depth image paths
        self.image_depth = np.asarray(self.data_info.iloc[:, 4])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        left_image_name = self.image_left[index]
        right_image_name = self.image_right[index]
        disparity_image_name = self.image_disparity[index]
        depth_image_name = self.image_depth[index]
        
        # Open image
        left_as_img = Image.open(left_image_name).convert("L")
        right_as_img = Image.open(right_image_name).convert("L")
        disparity_as_img = Image.open(disparity_image_name).convert("L")
        depth_as_img = Image.open(depth_image_name).convert("L")
        
        # Transform image to tensor
        left_as_tensor = self.to_tensor(left_as_img)
        right_as_tensor = self.to_tensor(right_as_img)
        disparity_as_tensor = self.to_tensor(disparity_as_img)
        depth_as_tensor = self.to_tensor(depth_as_img)
        # Get label(class) of the image based on the cropped pandas column
        #single_image_label = self.label_arr[index]

        return (left_as_tensor, right_as_tensor, disparity_as_tensor, depth_as_tensor)

    def __len__(self):
        return self.data_len
    
#transformations = transforms.Compose([transforms.ToTensor()])

#fire_datasets =  CustomDatasetFromImages('C:/Users/elektro/Desktop/datasets.csv')
        
#fire_dataset_loader = torch.utils.data.DataLoader(dataset=fire_datasets,
#                                                    batch_size=100,
#                                                    shuffle=False)

def CustomSplitLoader(datasets, batch_size, train_percentage, test_percentage, valid_percentage):
    # Split the datasets into training, testing, and validation
    num_train = len(datasets)
    indices = list(range(num_train))
    split_test = int(np.floor(train_percentage/100 * num_train))
    split_valid = int(np.floor(valid_percentage/100 * num_train))
    train_indices, test_idx = indices[split_test:], indices[:split_test]
    train_idx, valid_idx = train_indices[split_valid:], train_indices[:split_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # create train, validation and test loader
    train_loader = torch.utils.data.DataLoader(datasets, batch_size,
                    sampler=train_sampler, num_workers=4, pin_memory=False,)
    test_loader = torch.utils.data.DataLoader(datasets, batch_size,
                    sampler=test_sampler, num_workers=4, pin_memory=False,)
    valid_loader = torch.utils.data.DataLoader(datasets, batch_size,
                    sampler=valid_sampler, num_workers=4, pin_memory=False,)
    
    return train_loader, test_loader, valid_loader
    