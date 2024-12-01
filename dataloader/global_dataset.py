## Get dataset from kaggle if not already downloaded

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import pydicom
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sys

class_dict = {
    'Normal': 0,
    'Tuberculosis': 1,
    'mini_TB': 1,
    'mini_Normal': 0
}

class TBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset loader for TB Chest Radiography Database
        Args:
            data_dir (str): Path to the TB dataset directory
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                              std=[0.229])
        ])
        
        # Load metadata from xlsx files
        normal_df = pd.read_excel(os.path.join(data_dir, 'Normal.metadata.xlsx'))
        tb_df = pd.read_excel(os.path.join(data_dir, 'Tuberculosis.metadata.xlsx'))
        
        # Combine and create labels
        normal_df['label'] = 0  # Normal
        tb_df['label'] = 1      # Tuberculosis
        self.metadata = pd.concat([normal_df, tb_df], ignore_index=True)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if 'NORMAL'.lower() in row['FILE NAME'].lower():
            img_path = os.path.join(self.data_dir, 'Normal', row['FILE NAME']+'.'+row['FORMAT'].lower())
        else:
            img_path = os.path.join(self.data_dir, 'Tuberculosis', row['FILE NAME']+'.'+row['FORMAT'].lower())
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, row['label']

class IndianDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset loader for Indian chest X-ray DICOM files
        Args:
            data_dir (str): Path to the Indian dataset directory
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Get all DICOM files
        self.image_paths = []
        self.labels = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith('.dicom'):
                    self.image_paths.append(os.path.join(root, file))
                    # Extract label from parent directory name
                    label = os.path.basename(root)
                    self.labels.append(label)
        # Create label encoder
        self.unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load DICOM file
        dcm_path = self.image_paths[idx]
        dcm = pydicom.dcmread(dcm_path)
        
        # Convert to float and normalize
        image = dcm.pixel_array.astype(float)
        
        # Normalize to 0-1 range
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        return image, label_idx

def get_data_loaders(tb_data_path, indian_data_path, batch_size=32, train_val_split=0.8):
    """
    Creates stratified train and validation data loaders for both datasets
    
    Args:
        tb_data_path (str): Path to the TB dataset directory
        indian_data_path (str): Path to the Indian dataset directory
        batch_size (int): Batch size for the data loaders
        train_val_split (float): Proportion of data to use for training
        
    Returns:
        dict: Dictionary containing DataLoader objects for both datasets
    """
    print("Creating datasets")
    # Create datasets
    tb_dataset = TBDataset(tb_data_path)
    indian_dataset = IndianDataset(indian_data_path)

    print("Getting labels for stratification")
    # Get labels for stratification
    tb_labels = tb_dataset.metadata['label']
    indian_labels = [class_dict[label] for label in indian_dataset.labels]
    
    print("Creating stratified split indices")
    # Create stratified split indices
    tb_splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_val_split)
    indian_splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_val_split)
    
    # Get train/val indices while preserving class distribution
    tb_train_idx, tb_val_idx = next(tb_splitter.split(np.zeros(len(tb_labels)), tb_labels))
    indian_train_idx, indian_val_idx = next(indian_splitter.split(np.zeros(len(indian_labels)), indian_labels))
    
    # Create subset datasets using the indices
    tb_train_dataset = torch.utils.data.Subset(tb_dataset, tb_train_idx)
    tb_val_dataset = torch.utils.data.Subset(tb_dataset, tb_val_idx)
    indian_train_dataset = torch.utils.data.Subset(indian_dataset, indian_train_idx)
    indian_val_dataset = torch.utils.data.Subset(indian_dataset, indian_val_idx)
    
    # Create data loaders (rest remains the same)
    loaders = {
        'tb_train': DataLoader(tb_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'tb_val': DataLoader(tb_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'indian_train': DataLoader(indian_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'indian_val': DataLoader(indian_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'indian_whole': DataLoader(indian_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'global_whole': DataLoader(tb_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    return loaders

# Sample usage
def main():
    # Define paths to datasets
    tb_data_path = "data/TB_Chest_Radiography_Database"
    indian_data_path = "data/indian_dataset"
    
    # Get data loaders
    loaders = get_data_loaders(tb_data_path, indian_data_path)
    
    # Example of iterating through the data
    for batch_idx, (images, labels) in enumerate(loaders['tb_train']):
        print(f"Batch {batch_idx}")
        print(f"Image batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Process your batch here
        if batch_idx == 0:  # Just show first batch as example
            break
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"TB Train: {len(loaders['tb_train'].dataset)}")
    print(f"TB Val: {len(loaders['tb_val'].dataset)}")
    print(f"Indian Train: {len(loaders['indian_train'].dataset)}")
    print(f"Indian Val: {len(loaders['indian_val'].dataset)}")

if __name__ == "__main__":
    main()



