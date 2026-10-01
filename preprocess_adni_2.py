"""
ADNI MRI Preprocessing Script 2: Slice Cropping

This script preprocesses ADNI MRI scans by:
1. Loading original MRI images (256, 256, 170)
2. Cropping to informative slices (30-129) -> (100, 256, 170)
3. Saving to output directory with same structure

This is a simpler preprocessing that only does slice cropping,
preserving the original resolution in the other dimensions.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ADNI MRI data - Slice Cropping')
    parser.add_argument('--input_dir', type=str, default='/scratch/eslami/datasets/ADNI_Merged',
                        help='Input directory with raw ADNI data')
    parser.add_argument('--output_dir', type=str, default='/scratch/eslami/datasets/ADNI_Resized',
                        help='Output directory for preprocessed data')
    parser.add_argument('--crop_slices', type=int, nargs=2, default=[30, 130],
                        metavar=('START', 'END'),
                        help='Slice range to keep (default: 30 130, keeps slices 30-129)')
    parser.add_argument('--csv_path', type=str, 
                        default='Datasets/ADNI1_Combined_Unique_ImageDataID.csv',
                        help='CSV file with metadata')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize intensity values (zero-mean, unit-variance)')
    
    args = parser.parse_args()
    
    # Auto-load settings from config.yaml if it exists
    if os.path.exists('config.yaml'):
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'data' in config:
            args.input_dir = config['data'].get('data_dir', args.input_dir)
            args.csv_path = config['data'].get('csv_path', args.csv_path)
            if 'crop_slices' in config['data']:
                args.crop_slices = config['data']['crop_slices']
        
        print("Loaded preprocessing settings from config.yaml")
    
    return args


def normalize_intensity(img_data):
    """
    Normalize image intensity to zero mean and unit variance.
    Only considers non-zero voxels for statistics.
    """
    non_zero_mask = img_data > 0
    if non_zero_mask.any():
        mean_val = img_data[non_zero_mask].mean()
        std_val = img_data[non_zero_mask].std()
        if std_val > 0:
            img_data = np.where(non_zero_mask, (img_data - mean_val) / std_val, 0)
    return img_data


def crop_slices(img_data, start_slice, end_slice):
    """
    Crop slices from the first dimension.
    
    Args:
        img_data: 3D numpy array (D, H, W)
        start_slice: Start slice index (inclusive)
        end_slice: End slice index (exclusive)
    
    Returns:
        Cropped image with shape (end_slice - start_slice, H, W)
    """
    if len(img_data.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {img_data.shape}")
    
    if img_data.shape[0] < end_slice:
        raise ValueError(f"Cannot crop slices {start_slice}-{end_slice}, image has only {img_data.shape[0]} slices")
    
    # Crop the first dimension (slice dimension)
    cropped = img_data[start_slice:end_slice, :, :]
    
    return cropped


def process_single_image(input_path, output_path, start_slice, end_slice, normalize=False):
    """
    Process a single MRI image: load, crop slices, optionally normalize, and save.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save output NIfTI file
        start_slice: Start slice index for cropping
        end_slice: End slice index for cropping
        normalize: Whether to normalize intensity
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load NIfTI file
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Ensure correct dtype
        data = data.astype(np.float32)
        
        # Crop slices
        cropped_data = crop_slices(data, start_slice, end_slice)
        
        # Normalize if requested
        if normalize:
            cropped_data = normalize_intensity(cropped_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create new NIfTI image with identity affine
        # Use the original header but update the shape
        new_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
        nib.save(new_img, output_path)
        
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def load_metadata(csv_path, data_dir):
    """
    Load metadata from CSV and verify file existence.
    
    Returns:
        DataFrame with file paths
    """
    print(f"Loading metadata from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove rows with missing critical information
    df = df.dropna(subset=['Subject', 'Age', 'Group', 'Image Data ID'])
    
    # Verify that corresponding files exist
    print("Verifying file existence...")
    valid_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        subject_id = row['Subject']
        image_id = str(row['Image Data ID']).strip()
        
        # Construct expected file path
        subject_dir = os.path.join(data_dir, subject_id)
        image_dir = os.path.join(subject_dir, image_id)
        
        # Check if image directory exists
        if os.path.exists(image_dir):
            # Find .nii file in the directory
            nii_files = [f for f in os.listdir(image_dir) if f.endswith('.nii')]
            if len(nii_files) > 0:
                row['file_path'] = os.path.join(image_dir, nii_files[0])
                row['output_subject_dir'] = os.path.join(subject_dir.replace(data_dir, '').lstrip('/'), subject_id)
                row['output_image_dir'] = image_id
                valid_rows.append(row)
    
    df = pd.DataFrame(valid_rows)
    print(f"Found {len(df)} valid MRI scans with existing files")
    
    return df


def main():
    args = parse_args()
    
    print("\n")
    print("ADNI MRI Preprocessing - Slice Cropping")
    print("\n")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Crop slices: {args.crop_slices[0]}-{args.crop_slices[1]-1} (keeping {args.crop_slices[1] - args.crop_slices[0]} slices)")
    print(f"Normalize intensity: {args.normalize}")
    print(f"Expected output shape: ({args.crop_slices[1] - args.crop_slices[0]}, 256, 170)")
    print("\n")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    df = load_metadata(args.csv_path, args.input_dir)
    
    if len(df) == 0:
        print("No valid images found. Exiting.")
        return
    
    # Process images
    print(f"\nProcessing {len(df)} images...")
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        input_path = row['file_path']
        subject_id = row['Subject']
        image_id = str(row['Image Data ID']).strip()
        
        # Construct output path (maintain same directory structure)
        output_subject_dir = os.path.join(args.output_dir, subject_id)
        output_image_dir = os.path.join(output_subject_dir, image_id)
        os.makedirs(output_image_dir, exist_ok=True)
        
        # Find the input filename
        input_filename = os.path.basename(input_path)
        output_path = os.path.join(output_image_dir, input_filename)
        
        # Process image
        if process_single_image(input_path, output_path, 
                               args.crop_slices[0], args.crop_slices[1], 
                               args.normalize):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n")
    print("Preprocessing Complete!")
    print("\n")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Output directory: {args.output_dir}")
    print(f"Output shape: ({args.crop_slices[1] - args.crop_slices[0]}, 256, 170)")
    print("\n")
    
    # Save a copy of the CSV to the output directory for reference
    output_csv_path = os.path.join(args.output_dir, 'metadata.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Metadata saved to: {output_csv_path}")


if __name__ == "__main__":
    main()

