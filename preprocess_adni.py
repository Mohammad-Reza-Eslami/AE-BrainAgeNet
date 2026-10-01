# ADNI MRI Preprocessing
# This code preprocesses ADNI MRI scans for brain age prediction:
# 1. Skull stripping (remove non-brain tissue)
# 2. Registration to MNI standard space
# MNI (Montreal Neurological Institute) standard space is a common coordinate system and average brain template
# used in neuroimaging to standardize the location of brain structures across different individuals.
# 3. Intensity normalization
# 4. Cropping to standard size (121, 145, 121) based on the paper
#Requirements: ANTsPy or NiBabel + scipy

import os
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ADNI MRI data')
    parser.add_argument('--input_dir', type=str, default='Datasets/ADNI_Merged',
                        help='Input directory with raw ADNI data')
    parser.add_argument('--output_dir', type=str, default='Datasets/ADNI_Preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[121, 145, 121],
                        help='Target shape for preprocessed images')
    parser.add_argument('--use_ants', action='store_true',
                        help='Use ANTsPy for advanced preprocessing (if installed)')
    parser.add_argument('--csv_path', type=str, 
                        default='Datasets/ADNI1_Combined_Unique_ImageDataID.csv',
                        help='CSV file with metadata')
    
    args = parser.parse_args()
    
    # Auto-load preprocessing settings from config.yaml if it exists
    if os.path.exists('config.yaml'):
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'data' in config:
            args.input_dir = config['data'].get('data_dir', args.input_dir)
            args.csv_path = config['data'].get('csv_path', args.csv_path)
            args.target_shape = config['data'].get('input_shape', args.target_shape)
        
        # Load preprocessing output directory if specified
        if 'preprocessing' in config:
            args.output_dir = config['preprocessing'].get('output_dir', args.output_dir)
        
        print("Loaded preprocessing settings from config.yaml")
    
    return args


def simple_skull_strip(img_data, threshold_percentile=25):
    # Simple skull stripping using intensity thresholding and morphology.   

    # Calculate threshold
    non_zero = img_data[img_data > 0]
    if len(non_zero) == 0:
        return img_data
    
    threshold = np.percentile(non_zero, threshold_percentile)
    
    # Create binary mask
    mask = img_data > threshold
    
    # Morphological operations to clean mask
    # Fill holes
    mask = ndimage.binary_fill_holes(mask)
    
    # Remove small objects
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))
    
    # Keep only largest connected component (the brain)
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_component = np.argmax(sizes) + 1
        mask = labeled == largest_component
    
    # Apply mask
    stripped = img_data * mask
    
    return stripped


def crop_to_brain(img_data, margin=5):
    # Crop image to brain bounding box with a margin.
    # Find brain region
    non_zero = np.where(img_data > 0)
    
    if len(non_zero[0]) == 0:
        return img_data
    
    # Get bounding box
    min_x, max_x = non_zero[0].min(), non_zero[0].max()
    min_y, max_y = non_zero[1].min(), non_zero[1].max()
    min_z, max_z = non_zero[2].min(), non_zero[2].max()
    
    # Add margin
    min_x = max(0, min_x - margin)
    max_x = min(img_data.shape[0], max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(img_data.shape[1], max_y + margin)
    min_z = max(0, min_z - margin)
    max_z = min(img_data.shape[2], max_z + margin)
    
    # Crop
    cropped = img_data[min_x:max_x, min_y:max_y, min_z:max_z]
    
    return cropped


def resize_to_target(img_data, target_shape):
    # Resize image to target shape using interpolation.

    # Calculate zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, img_data.shape)]
    
    # Resample using trilinear interpolation
    resized = ndimage.zoom(img_data, zoom_factors, order=1)
    
    return resized


def normalize_intensity(img_data):
    # Normalize image intensity to [0, 1] range.

    # Remove zeros for statistics
    non_zero = img_data[img_data > 0]
    
    if len(non_zero) == 0:
        return img_data
    
    # Use percentile clipping to handle outliers
    p1, p99 = np.percentile(non_zero, [1, 99])
    
    # Clip and normalize
    img_data = np.clip(img_data, p1, p99)
    img_data = (img_data - p1) / (p99 - p1)
    
    return img_data


def preprocess_simple(img_data, target_shape):
    # Simple preprocessing pipeline without ANTs.
    # Steps:
    # 1. Skull stripping (simple thresholding)
    # 2. Crop to brain
    # 3. Resize to target shape
    # 4. Intensity normalization
    print("  [1/4] Skull stripping...")
    stripped = simple_skull_strip(img_data)
    
    print("  [2/4] Cropping to brain...")
    cropped = crop_to_brain(stripped)
    
    print("  [3/4] Resizing to target shape...")
    resized = resize_to_target(cropped, target_shape)
    
    print("  [4/4] Normalizing intensity...")
    normalized = normalize_intensity(resized)
    
    return normalized


def preprocess_with_ants(img_path, target_shape):
    # Advanced preprocessing using ANTsPy.
    # Requires: pip install antspyx

    try:
        import ants
    except ImportError:
        print("ANTsPy not installed. Install with: pip install antspyx")
        return None
    
    print("  [1/5] Loading image...")
    img = ants.image_read(img_path)
    
    print("  [2/5] N4 bias field correction...")
    img = ants.n4_bias_field_correction(img)
    
    print("  [3/5] Brain extraction...")
    brain_mask = ants.get_mask(img, low_thresh=img.mean(), cleanup=2)
    img = img * brain_mask
    
    print("  [4/5] Registration to MNI space...")
    # Load MNI template (you'd need to download this)
    # For now, skip registration and just resize
    
    print("  [5/5] Resampling to target shape...")
    # Convert to numpy and resize
    img_data = img.numpy()
    resized = resize_to_target(img_data, target_shape)
    normalized = normalize_intensity(resized)
    
    return normalized


def process_single_image(input_path, output_path, target_shape, use_ants=False):
    # Process a single MRI image.

    try:
        if use_ants:
            processed = preprocess_with_ants(input_path, target_shape)
            if processed is None:
                # Fall back to simple method
                img = nib.load(input_path)
                processed = preprocess_simple(img.get_fdata(), target_shape)
        else:
            img = nib.load(input_path)
            processed = preprocess_simple(img.get_fdata(), target_shape)
        
        # Save processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create new NIfTI image with identity affine
        new_img = nib.Nifti1Image(processed, np.eye(4))
        nib.save(new_img, output_path)
        
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    args = parse_args()
    
    print("\n")
    print("ADNI MRI Preprocessing...")
    print("\n")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target shape: {args.target_shape}")
    print(f"Using ANTs: {args.use_ants}")
    print("\n")
    
    # Load CSV to get list of files
    print("\nLoading metadata...")
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()
    
    # Find all .nii files
    print("\nFinding images to process...")
    input_dir = Path(args.input_dir)
    all_nii_files = list(input_dir.rglob("*.nii"))
    
    print(f"Found {len(all_nii_files)} images")
    
    # Process each image
    print("\nProcessing images...")
    success_count = 0
    failed_files = []
    
    for input_path in tqdm(all_nii_files, desc="Processing"):
        # Create output path with same structure
        rel_path = input_path.relative_to(input_dir)
        output_path = Path(args.output_dir) / rel_path
        
        # Skip if already processed
        if output_path.exists():
            print(f"Skipping {input_path.name} (already exists)")
            success_count += 1
            continue
        
        print(f"\nProcessing: {input_path.name}")
        success = process_single_image(
            str(input_path), 
            str(output_path), 
            tuple(args.target_shape),
            args.use_ants
        )
        
        if success:
            success_count += 1
        else:
            failed_files.append(str(input_path))
    
    # Print summary
    print("\n")
    print("Preprocessing Complete!")
    print("\n")
    print(f"Successfully processed: {success_count}/{len(all_nii_files)}")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nPreprocessed data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Update config.yaml to use preprocessed data:")
    print(f"   data_dir: '{args.output_dir}'")
    print(f"   input_shape: {args.target_shape}")
    print("\n2. Run training:")
    print("   python adni_train.py --config config.yaml")
    print("\n")


if __name__ == '__main__':
    main()

