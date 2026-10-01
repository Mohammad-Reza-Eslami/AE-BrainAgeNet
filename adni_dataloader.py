#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ADNI Dataset Dataloader with Subject-Level Splitting

import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ADNIDataLoader:
    # It handles subject-level splitting.
    # Each subject can have multiple MRI scans, so we ensure that all scans-
    # from the same subject are kept together in the same split (train/val/test).

    
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
        
        # Create subject-level splits
        self.train_subjects, self.val_subjects, self.test_subjects = self._split_subjects()
        
        print(f"Dataset loaded successfully!")
        print(f"Total subjects: {len(self.df['Subject'].unique())}")
        print(f"Total scans: {len(self.df)}")
        print(f"Train subjects: {len(self.train_subjects)}")
        print(f"Val subjects: {len(self.val_subjects)}")
        print(f"Test subjects: {len(self.test_subjects)}")
        
    def _load_metadata(self):
        # Load and process the CSV metadata file.
        print(f"Loading metadata from {self.csv_path}...")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Remove rows with missing critical information (Just in case to be standard for later use)
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
    
    def _split_subjects(self):
        # Split subjects (not individual scans) into train/val/test sets.
        # Get unique subjects
        subjects = self.df['Subject'].unique()
        
        # For stratified splitting, we'll use the most common diagnosis per subject
        subject_groups = {}
        for subject in subjects:
            subject_data = self.df[self.df['Subject'] == subject]
            # Get the most common group for this subject
            most_common_group = subject_data['Group'].mode()[0]
            subject_groups[subject] = most_common_group
        
        # Convert to lists for stratification
        subjects_list = list(subjects)
        groups_list = [subject_groups[s] for s in subjects_list]
        
        # First split: separate test set
        train_val_subjects, test_subjects, train_val_groups, _ = train_test_split(
            subjects_list, groups_list, 
            test_size=self.test_ratio,
            stratify=groups_list,
            random_state=self.random_state
        )
        
        # Second split: separate train and val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_subjects, val_subjects, _, _ = train_test_split(
            train_val_subjects, train_val_groups,
            test_size=val_ratio_adjusted,
            stratify=train_val_groups,
            random_state=self.random_state
        )
        
        return train_subjects, val_subjects, test_subjects
    
    def _load_nifti(self, file_path):
        # Load and preprocess a NIfTI file.
        try:
            # Load NIfTI file
            img = nib.load(file_path) # load the image as numpy array
            data = img.get_fdata() # converts to floating point data
            
            # Handle NaN values (just in case to be standard for later use)
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
        # Resize image to target shape by center cropping or padding.
        current_shape = image.shape
        
        # Calculate padding/cropping for each dimension
        result = np.zeros(target_shape)
        
        slices_current = []
        slices_result = []
        
        for i in range(3): # the methode that they used in their paper
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
        # Load data for a specific subset (train/val/test).

        # Select appropriate subject list
        if subset == 'train':
            subjects = self.train_subjects
        elif subset == 'val':
            subjects = self.val_subjects
        elif subset == 'test':
            subjects = self.test_subjects
        else:
            raise ValueError("subset must be 'train', 'val', or 'test'")
        
        # Filter dataframe for selected subjects
        subset_df = self.df[self.df['Subject'].isin(subjects)].copy()
        
        print(f"\nLoading {subset} data...")
        print(f"Number of scans in {subset} set: {len(subset_df)}")
        
        # Load all images
        X_list = []
        y_list = []
        valid_indices = []
        
        for idx, row in subset_df.iterrows():
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
        # Get dataset statistics.
        stats = {
            'total_subjects': len(self.df['Subject'].unique()),
            'total_scans': len(self.df),
            'train_subjects': len(self.train_subjects),
            'val_subjects': len(self.val_subjects),
            'test_subjects': len(self.test_subjects),
            'diagnosis_distribution': self.df['Group'].value_counts().to_dict(),
        }
        return stats
    
    def print_statistics(self):
        # Print detailed dataset statistics.
        stats = self.get_statistics()
        
        print("\n")
        print("ADNI Dataset Statistics:")
        print(f"Total subjects: {stats['total_subjects']}")
        print(f"Total scans: {stats['total_scans']}")
        print(f"\nSubject split:")
        print(f"  Train: {stats['train_subjects']}")
        print(f"  Val:   {stats['val_subjects']}")
        print(f"  Test:  {stats['test_subjects']}")
        print(f"\nDiagnosis distribution:")
        for diagnosis, count in stats['diagnosis_distribution'].items():
            print(f"  {diagnosis}: {count}")
        print("\n")


def test_dataloader():
    # Test the dataloader functionality.
    data_dir = 'Datasets/ADNI_Merged'
    csv_path = 'Datasets/ADNI1_Combined_Unique_ImageDataID.csv'
    
    # Initialize dataloader
    dataloader = ADNIDataLoader(
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

