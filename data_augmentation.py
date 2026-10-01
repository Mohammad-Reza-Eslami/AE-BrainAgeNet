import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate, shift, zoom, gaussian_filter
from scipy.ndimage import map_coordinates
import random


class Augmentation3D:
    
    def __init__(self, 
                 rotation_range=10.0,  # ±10 degrees as suggested
                 translation_range=0.05,  # 5% of image size
                 scale_range=0.1,  # ±10% scaling
                 intensity_shift_range=0.1,  # ±10% intensity shift
                 intensity_scale_range=0.1,  # ±10% intensity scaling
                 flip_prob=0.5,  # 50% chance of flipping
                 elastic_alpha=None,  # Elastic deformation strength (None = disabled)
                 elastic_sigma=None,  # Elastic deformation smoothness
                 p=0.5):  # Probability of applying each augmentation

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.flip_prob = flip_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p = p
    
    def rotate(self, image, axes=(1, 2)):

        if random.random() > self.p:
            return image
        
        # Random rotation angle between -rotation_range and +rotation_range
        # User suggested ±10 degrees, which we implement here
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # Randomly choose one axis to rotate around (more conservative than rotating around all)
        # This prevents over-augmentation while still providing diversity
        axis_choice = random.choice(axes) if isinstance(axes, (list, tuple)) else axes
        
        if axis_choice == 1:
            # Rotate around axis 1 (sagittal plane - left/right rotation)
            image = rotate(image, angle, axes=(1, 2), reshape=False, 
                          order=1, mode='constant', cval=0.0)
        elif axis_choice == 2:
            # Rotate around axis 2 (coronal plane - front/back rotation)
            image = rotate(image, angle, axes=(0, 2), reshape=False,
                          order=1, mode='constant', cval=0.0)
        elif axis_choice == 0:
            # Rotate around axis 0 (axial plane - top/bottom rotation)
            image = rotate(image, angle, axes=(0, 1), reshape=False,
                          order=1, mode='constant', cval=0.0)
        
        return image
    
    def translate(self, image):
        # Apply random translation
        if random.random() > self.p:
            return image
        
        # Calculate maximum translation in pixels
        max_shift = [int(s * self.translation_range) for s in image.shape]
        
        # Random shifts for each dimension
        shifts = [random.randint(-max_shift[i], max_shift[i]) 
                 for i in range(len(image.shape))]
        
        # Apply translation
        image = shift(image, shifts, order=1, mode='constant', cval=0.0)
        
        return image
    
    def scale(self, image):
        # Apply random scaling (zoom)
        if random.random() > self.p:
            return image
        
        # Store original shape
        original_shape = image.shape
        
        # Random scale factor
        scale_factor = 1.0 + random.uniform(-self.scale_range, self.scale_range)
        
        # Apply zoom
        zoomed = zoom(image, scale_factor, order=1, mode='constant', cval=0.0)
        
        # Crop or pad to original size
        if scale_factor > 1.0:
            # Zoomed in - need to crop to center
            crop_start = [(zoomed.shape[i] - original_shape[i]) // 2 
                         for i in range(len(original_shape))]
            crop_end = [crop_start[i] + original_shape[i] 
                       for i in range(len(original_shape))]
            image = zoomed[crop_start[0]:crop_end[0],
                          crop_start[1]:crop_end[1],
                          crop_start[2]:crop_end[2]]
        elif scale_factor < 1.0:
            # Zoomed out - need to pad
            pad_before = [(original_shape[i] - zoomed.shape[i]) // 2 
                         for i in range(len(original_shape))]
            pad_after = [original_shape[i] - zoomed.shape[i] - pad_before[i]
                        for i in range(len(original_shape))]
            image = np.pad(zoomed, 
                          [(pad_before[i], pad_after[i]) for i in range(len(original_shape))],
                          mode='constant', constant_values=0.0)
        else:
            image = zoomed
        
        # Ensure exact original shape (handle any rounding errors)
        if image.shape != original_shape:
            result = np.zeros(original_shape, dtype=image.dtype)
            slices = [slice(0, min(original_shape[i], image.shape[i])) 
                     for i in range(len(original_shape))]
            result[slices[0], slices[1], slices[2]] = image[slices[0], slices[1], slices[2]]
            return result
        
        return image
    
    def adjust_intensity(self, image):
        # Apply random intensity adjustments
        if random.random() > self.p:
            return image
        
        # Intensity shift (brightness)
        shift_amount = random.uniform(-self.intensity_shift_range, 
                                     self.intensity_shift_range)
        image = image + shift_amount
        
        # Intensity scaling (contrast)
        scale_amount = 1.0 + random.uniform(-self.intensity_scale_range,
                                            self.intensity_scale_range)
        image = image * scale_amount
        
        return image
    
    def flip(self, image):
        # Randomly flip along axes
        # Flip along depth axis (left-right)
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=0)
        
        # Note: We typically don't flip along other axes for brain images
        # as it would change anatomical orientation
        
        return image
    
    def elastic_deformation(self, image):
        # Apply elastic deformation
        if self.elastic_alpha is None or random.random() > self.p:
            return image
        
        shape = image.shape
        alpha = self.elastic_alpha * shape[0]  # Scale by image size
        sigma = self.elastic_sigma
        
        # Generate random displacement fields
        dx = gaussian_filter((random.random() * 2 - 1) * alpha, sigma, mode='constant')
        dy = gaussian_filter((random.random() * 2 - 1) * alpha, sigma, mode='constant')
        dz = gaussian_filter((random.random() * 2 - 1) * alpha, sigma, mode='constant')
        
        # Create coordinate grids
        x, y, z = np.meshgrid(np.arange(shape[0]), 
                             np.arange(shape[1]),
                             np.arange(shape[2]), indexing='ij')
        
        # Apply displacements
        indices = np.reshape(x + dx, (-1, 1)), \
                  np.reshape(y + dy, (-1, 1)), \
                  np.reshape(z + dz, (-1, 1))
        
        # Sample image at new coordinates
        image = map_coordinates(image, indices, order=1, mode='constant', cval=0.0)
        image = image.reshape(shape)
        
        return image
    
    def __call__(self, image):

        # Make a copy to avoid modifying original
        augmented = image.copy()
        
        # Apply augmentations in a reasonable order
        # 1. Geometric transformations first
        augmented = self.rotate(augmented)
        augmented = self.translate(augmented)
        augmented = self.scale(augmented)
        augmented = self.flip(augmented)
        
        # 2. Elastic deformation (if enabled)
        if self.elastic_alpha is not None:
            augmented = self.elastic_deformation(augmented)
        
        # 3. Intensity adjustments last
        augmented = self.adjust_intensity(augmented)
        
        return augmented


class AugmentedTensorDataset(Dataset):
    
    def __init__(self, X, y, mode='train', augmentation=None):

        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.FloatTensor(y) if not isinstance(y, torch.Tensor) else y
        self.mode = mode
        self.augmentation = augmentation if mode == 'train' else None
        
        # Handle different input formats
        if len(self.X.shape) == 4:  # (N, D, H, W) - add channel dimension
            self.X = self.X.unsqueeze(1)  # (N, 1, D, H, W)
        elif len(self.X.shape) == 5:  # (N, C, D, H, W)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {self.X.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # Apply augmentation only during training
        if self.augmentation is not None and self.mode == 'train':
            # Convert to numpy for augmentation (remove channel dim temporarily)
            if x.dim() == 4:  # (C, D, H, W)
                x_np = x[0].numpy()  # (D, H, W)
            else:
                x_np = x.numpy()
            
            # Apply augmentation
            x_np = self.augmentation(x_np)
            
            # Ensure array is contiguous (fixes negative stride issue)
            # This is necessary because some augmentations (like flip) can create views with negative strides
            x_np = np.ascontiguousarray(x_np)
            
            # Convert back to tensor and restore channel dimension
            x = torch.FloatTensor(x_np).unsqueeze(0)  # (1, D, H, W)
        
        return x, y


def get_augmentation(config='standard'):

    if config == 'light':
        # Minimal: Just rotation as user suggested
        return Augmentation3D(
            rotation_range=10.0,
            translation_range=0.0,  # Disabled
            scale_range=0.0,  # Disabled
            intensity_shift_range=0.0,  # Disabled
            intensity_scale_range=0.0,  # Disabled
            flip_prob=0.0,  # Disabled
            p=0.5
        )
    elif config == 'standard':
        # Standard: Rotation + translation + intensity adjustments
        return Augmentation3D(
            rotation_range=10.0,  # ±10 degrees as user suggested
            translation_range=0.05,  # 5% translation
            scale_range=0.05,  # 5% scaling (conservative)
            intensity_shift_range=0.1,  # 10% intensity shift
            intensity_scale_range=0.1,  # 10% intensity scaling
            flip_prob=0.3,  # 30% chance of flipping
            p=0.5  # 50% chance of applying each augmentation
        )
    elif config == 'heavy':
        # Heavy: All augmentations including elastic deformation
        return Augmentation3D(
            rotation_range=10.0,
            translation_range=0.1,  # 10% translation
            scale_range=0.1,  # 10% scaling
            intensity_shift_range=0.15,  # 15% intensity shift
            intensity_scale_range=0.15,  # 15% intensity scaling
            flip_prob=0.5,
            elastic_alpha=10.0,  # Enable elastic deformation
            elastic_sigma=3.0,
            p=0.6  # 60% chance of applying each augmentation
        )
    else:
        raise ValueError(f"Unknown augmentation config: {config}. Must be 'light', 'standard', or 'heavy'")



