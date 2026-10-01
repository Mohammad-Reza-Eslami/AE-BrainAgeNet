#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ADNI Dataset Dataloader with Scan-Level Splitting
# 
# This dataloader splits data at the scan level (not subject level).
# Individual scans can be split independently, even if they come from the same subject.
# This is useful for comparing results with subject-level splitting.

import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ADNIDataLoaderScanLevel:
    """
    ADNI Dataset Dataloader with Scan-Level Splitting
    
    Unlike ADNIDataLoader, this splits individual scans independently.
    Scans from the same subject can end up in different splits (train/val/test).
    This allows for comparison with subject-level splitting approaches.
    """
    
    def __init__(self, data_dir, csv_path, target_shape=(121, 145, 121), 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                 random_state=42, diagnosis_filter=None, use_original_size=False,
                 crop_slices=None):

        self.data_dir = data_dir
        self.csv_path = csv_path
        # Keep target_shape even when use_original_size=True to ensure shape consistency
        # We'll still resize/crop/pad if shapes don't match
        self.target_shape = target_shape
        self.use_original_size = use_original_size
        # crop_slices: tuple (start, end) to crop slices, e.g., (30, 130) keeps slices 30-129
        self.crop_slices = crop_slices
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.diagnosis_filter = diagnosis_filter
        
        # Load and process metadata
        self.df = self._load_metadata()
        
        # Create scan-level splits (not subject-level)
        self.train_indices, self.val_indices, self.test_indices = self._split_scans()
        
        print(f"Dataset loaded successfully!")
        print(f"Total subjects: {len(self.df['Subject'].unique())}")
        print(f"Total scans: {len(self.df)}")
        print(f"Train scans: {len(self.train_indices)}")
        print(f"Val scans: {len(self.val_indices)}")
        print(f"Test scans: {len(self.test_indices)}")
        
    def _load_metadata(self):
        """Load and process the CSV metadata file."""
        print(f"Loading metadata from {self.csv_path}...")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Remove rows with missing critical information
        df = df.dropna(subset=['Subject', 'Age', 'Group', 'Image Data ID'])
        
        # Filter by diagnosis if specified
        if self.diagnosis_filter is not None:
            df = df[df['Group'].isin(self.diagnosis_filter)]
            print(f"Filtered for diagnoses: {self.diagnosis_filter}")
        
        # Verify that corresponding files exist
        print("Verifying file existence...")
        valid_rows = []
        for idx, row in df.iterrows():
            subject_id = row['Subject']
            image_id = str(row['Image Data ID']).strip()
            
            # Construct expected file path
            subject_dir = os.path.join(self.data_dir, subject_id)
            image_dir = os.path.join(subject_dir, image_id)
            
            # Check if image directory exists
            if os.path.exists(image_dir):
                # Find .nii file in the directory
                nii_files = [f for f in os.listdir(image_dir) if f.endswith('.nii')]
                if len(nii_files) > 0:
                    row['file_path'] = os.path.join(image_dir, nii_files[0])
                    valid_rows.append(row)
        
        df = pd.DataFrame(valid_rows)
        print(f"Found {len(df)} valid MRI scans with existing files")
        
        return df
    
    def _split_scans(self):
        """
        Split individual scans (not subjects) into train/val/test sets.
        
        This is the key difference from ADNIDataLoader - we split at the scan level,
        so scans from the same subject can end up in different splits.
        """
        # Get all scan indices
        scan_indices = self.df.index.tolist()
        
        # Get diagnosis groups for stratified splitting
        groups = self.df['Group'].values
        
        # First split: separate test set
        train_val_indices, test_indices, train_val_groups, _ = train_test_split(
            scan_indices, groups,
            test_size=self.test_ratio,
            stratify=groups,
            random_state=self.random_state
        )
        
        # Second split: separate train and val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_indices, val_indices, _, _ = train_test_split(
            train_val_indices, train_val_groups,
            test_size=val_ratio_adjusted,
            stratify=train_val_groups,
            random_state=self.random_state
        )
        
        return train_indices, val_indices, test_indices
    
    def _load_nifti(self, file_path):
        """Load and preprocess a NIfTI file."""
        try:
            # Load NIfTI file
            img = nib.load(file_path)  # load the image as numpy array
            data = img.get_fdata()  # converts to floating point data
            
            # Handle NaN values
            data = np.nan_to_num(data, nan=0.0)
            
            # Crop slices if specified (remove non-informative slices)
            # crop_slices: (start, end) - keeps slices from start to end-1
            # For shape (256, 256, 170), cropping first dimension slices 30-129 gives (100, 256, 170)
            if self.crop_slices is not None:
                start_slice, end_slice = self.crop_slices
                if len(data.shape) == 3:
                    # Crop the first dimension (slice dimension)
                    # Original: (256, 256, 170) -> Cropped: (100, 256, 170) for slices 30-129
                    if data.shape[0] >= end_slice:
                        data = data[start_slice:end_slice, :, :]
                    else:
                        print(f"Warning: Cannot crop slices {start_slice}-{end_slice}, data shape is {data.shape}")
                else:
                    print(f"Warning: Unexpected data shape {data.shape}, skipping slice cropping")
            
            # Ensure consistent shape - always resize/crop/pad to target_shape if specified
            # Even with use_original_size=True, we need to ensure all images have the same shape
            if self.target_shape is not None:
                if data.shape != self.target_shape:
                    # Resize/crop/pad to match target shape
                    data = self._resize_image(data, self.target_shape)
            # If target_shape is None, keep original size - but this may cause shape inconsistencies
            
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _resize_image(self, image, target_shape):
        """Resize image to target shape by center cropping or padding."""
        current_shape = image.shape
        
        # Calculate padding/cropping for each dimension
        result = np.zeros(target_shape)
        
        slices_current = []
        slices_result = []
        
        for i in range(3):  # the method that they used in their paper
            if current_shape[i] > target_shape[i]:
                # Crop
                start = (current_shape[i] - target_shape[i]) // 2
                end = start + target_shape[i]
                slices_current.append(slice(start, end))
                slices_result.append(slice(0, target_shape[i]))
            else:
                # Pad
                start = (target_shape[i] - current_shape[i]) // 2
                end = start + current_shape[i]
                slices_current.append(slice(0, current_shape[i]))
                slices_result.append(slice(start, end))
        
        result[slices_result[0], slices_result[1], slices_result[2]] = image[slices_current[0], slices_current[1], slices_current[2]]
        
        return result
    
    def load_data(self, subset='train', normalize=True):
        """
        Load data for a specific subset (train/val/test).
        
        Args:
            subset: 'train', 'val', or 'test'
            normalize: Whether to normalize each image individually
        
        Returns:
            X: numpy array of images (N, D, H, W)
            y: numpy array of ages (N,)
            metadata: DataFrame with metadata for loaded scans
        """
        # Select appropriate indices
        if subset == 'train':
            indices = self.train_indices
        elif subset == 'val':
            indices = self.val_indices
        elif subset == 'test':
            indices = self.test_indices
        else:
            raise ValueError("subset must be 'train', 'val', or 'test'")
        
        # Filter dataframe for selected indices
        subset_df = self.df.loc[indices].copy()
        
        print(f"\nLoading {subset} data...")
        print(f"Number of scans in {subset} set: {len(subset_df)}")
        
        # Load all images
        X_list = []
        y_list = []
        valid_indices = []
        
        for idx in subset_df.index:
            row = subset_df.loc[idx]
            # Load MRI image
            img_data = self._load_nifti(row['file_path'])
            
            if img_data is not None:
                X_list.append(img_data)
                y_list.append(row['Age'])
                valid_indices.append(idx)
        
        # Convert to numpy arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            # Normalize each image individually
            for i in range(len(X)):
                img = X[i]
                # Remove zero padding from statistics
                non_zero_mask = img > 0
                if non_zero_mask.any():
                    mean_val = img[non_zero_mask].mean()
                    std_val = img[non_zero_mask].std()
                    if std_val > 0:
                        X[i] = np.where(non_zero_mask, (img - mean_val) / std_val, 0)
        
        # Get corresponding metadata
        metadata = subset_df.loc[valid_indices].copy()
        
        print(f"Loaded {len(X)} images successfully")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Age range: {y.min():.1f} - {y.max():.1f} years")
        print(f"Mean age: {y.mean():.1f} ± {y.std():.1f} years")
        
        return X, y, metadata
    
    def get_statistics(self):
        """Get dataset statistics."""
        train_subjects = set(self.df.loc[self.train_indices, 'Subject'].unique())
        val_subjects = set(self.df.loc[self.val_indices, 'Subject'].unique())
        test_subjects = set(self.df.loc[self.test_indices, 'Subject'].unique())
        
        stats = {
            'total_subjects': len(self.df['Subject'].unique()),
            'total_scans': len(self.df),
            'train_scans': len(self.train_indices),
            'val_scans': len(self.val_indices),
            'test_scans': len(self.test_indices),
            'train_subjects': len(train_subjects),
            'val_subjects': len(val_subjects),
            'test_subjects': len(test_subjects),
            'diagnosis_distribution': self.df['Group'].value_counts().to_dict(),
        }
        return stats
    
    def print_statistics(self):
        """Print detailed dataset statistics."""
        stats = self.get_statistics()
        
        print("\n")
        print("ADNI Dataset Statistics (Scan-Level Splitting):")
        print(f"Total subjects: {stats['total_subjects']}")
        print(f"Total scans: {stats['total_scans']}")
        print(f"\nScan split:")
        print(f"  Train: {stats['train_scans']} scans")
        print(f"  Val:   {stats['val_scans']} scans")
        print(f"  Test:  {stats['test_scans']} scans")
        print(f"\nDiagnosis distribution:")
        for diagnosis, count in stats['diagnosis_distribution'].items():
            print(f"  {diagnosis}: {count}")
        print("\n")


def test_dataloader():
    """Test the dataloader functionality."""
    data_dir = 'Datasets/ADNI_Merged'
    csv_path = 'Datasets/ADNI1_Combined_Unique_ImageDataID.csv'
    
    # Initialize dataloader
    dataloader = ADNIDataLoaderScanLevel(
        data_dir=data_dir,
        csv_path=csv_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # Print statistics
    dataloader.print_statistics()
    
    # Load a small subset for testing
    print("Testing data loading...")
    X_train, y_train, meta_train = dataloader.load_data('train')
    
    print(f"\nSuccessfully loaded {len(X_train)} training images!")
    print(f"Image shape: {X_train[0].shape}")
    print(f"First 5 ages: {y_train[:5]}")


if __name__ == '__main__':
    test_dataloader()

