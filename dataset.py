import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split

class VortexDataset(Dataset):
    def __init__(self, data_dir, split='train', input_length=10, feature_cols=None):
        """
        Args:
            data_dir (str): Directory containing .txt files.
            split (str): 'train', 'val', or 'test'.
            input_length (int): Number of time steps in input (e.g., 10).
            feature_cols (list): List of feature column names (e.g., Val5_1 to Val122_2).
        """
        self.input_length = input_length
        self.data_dir = data_dir
        self.split = split
        self.feature_cols = feature_cols
        
        # Load all files
        self.data_files = glob.glob(os.path.join(data_dir, '*.txt'))
        if not self.data_files:
            raise ValueError(f"No files found in {data_dir}")
        
        # Load and prepare data
        self.data_list = []
        self.labels_list = []
        self._load_data()
        
    def _load_data(self):
        # Collect all samples
        all_features = []
        all_labels = []
        
        for file_path in self.data_files:
            df = pd.read_csv(file_path, sep='\s+', header=0)
            
            # Select feature columns
            if self.feature_cols is None:
                all_feature_cols = [col for col in df.columns if col not in ['Track', 'Year', 'Month', 'Day', 'Label']]
                # Remove first 8 and last 200 feature columns
                self.feature_cols = all_feature_cols[8:-200]
                if len(self.feature_cols) != 284:
                    raise ValueError(f"Expected 284 feature columns, got {len(self.feature_cols)}")
            
            # Extract features and labels
            features = df[self.feature_cols].values  # Shape: (n_rows, 282)
            labels = df['Label'].values  # Shape: (n_rows,)
            
            # Handle NaN
            features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))
            
            # Generate samples
            for i in range(len(features) - self.input_length):
                x = features[i:i+self.input_length]
                y = labels[i+self.input_length]
                all_features.append(x)
                all_labels.append(y)
        
        # Convert to numpy arrays
        all_features = np.array(all_features)  # Shape: (n_samples, input_length, 282)
        all_labels = np.array(all_labels)  # Shape: (n_samples,)
        
        # Stratified split
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            range(len(all_features)), all_labels,
            test_size=0.3, stratify=all_labels, random_state=42
        )
        val_idx, test_idx, val_labels, test_labels = train_test_split(
            temp_idx, temp_labels,
            test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        # Select split
        if self.split == 'train':
            indices = train_idx
        elif self.split == 'val':
            indices = val_idx
        else:  # test
            indices = test_idx
        
        # Store selected samples
        self.data_list = [all_features[i] for i in indices]
        self.labels_list = [all_labels[i] for i in indices]
        
        # Log label distribution
        if len(self.labels_list) > 0:
            label_ratio = sum(self.labels_list) / len(self.labels_list)
            print(f"{self.split.capitalize()} set label=1 ratio: {label_ratio:.4f}")
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data_list[idx], dtype=torch.float32)  # Shape: (input_length, 282)
        y = torch.tensor(self.labels_list[idx], dtype=torch.float32)  # Shape: ()
        return x, y
    
    def __len__(self):
        return len(self.data_list)