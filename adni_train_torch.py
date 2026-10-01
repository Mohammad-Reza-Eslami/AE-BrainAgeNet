# Training script for ADNI Brain Age Prediction using 3D CNNs with PyTorch

# With this code we can train various deep learning models (DenseNet, ResNet, VGG16, SFCN) from the 2022 paper
# and our custom models on the ADNI dataset for brain age prediction.
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import numpy as np
import pandas as pd
import random
import argparse
import json
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import wandb
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Import dataloader
from adni_dataloader import ADNIDataLoader # Subject-level splitting 
from adni_dataloader_scan_level import ADNIDataLoaderScanLevel # Scan-level splitting

# Import our custom models
from model.brainagenet import BrainAgeNet
from model.brainagenet_v2 import BrainAgeNetV2, ImprovedLoss

# Import data augmentation
from data_augmentation import AugmentedTensorDataset, get_augmentation


class TrainingMonitor:
    # Custom class to monitor training progress in PyTorch.
    def __init__(self, output_path):
        self.output_path = output_path
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    def update(self, epoch, train_loss, val_loss):
        self.history['loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['mae'].append(train_loss)
        self.history['val_mae'].append(val_loss)


def parse_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Train Brain Age Prediction Model on ADNI Dataset')
    
    # Config file option
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (overrides other arguments)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Datasets/ADNI_Merged', help='Path to ADNI_Merged directory')
    parser.add_argument('--csv_path', type=str, default='Datasets/ADNI1_Combined_Unique_ImageDataID.csv', help='Path to CSV file with metadata')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet', 'resnet', 'vgg16', 'sfcn', 'brainagenet', 'brainagenet_v2'],
                        help='Model architecture to use')
    parser.add_argument('--model_config', type=str, default='standard', choices=['light', 'standard', 'heavy'], help='Model configuration (for brainagenet_v2 only)')
    parser.add_argument('--use_improved_loss', action='store_true', help='Use improved loss (MAE + correlation) instead of plain MAE')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[121, 145, 121], help='Input shape for MRI images')
    parser.add_argument('--use_original_size', action='store_true', help='Use original MRI size (256, 256, 170) instead of resized. Requires raw data from ADNI_Merged directory.')
    parser.add_argument('--crop_slices', type=int, nargs=2, default=None, metavar=('START', 'END'),
                        help='Crop slices to keep only informative ones. E.g., --crop_slices 30 130 keeps slices 30-129. Reduces memory and focuses on relevant brain regions.')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs per iteration')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor per iteration')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'], help='Optimizer to use')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of epochs without improvement before stopping)')
    
    # Data augmentation parameters
    parser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation for training')
    parser.add_argument('--augmentation_config', type=str, default='standard', choices=['light', 'standard', 'heavy'],
                        help='Augmentation configuration: light (rotation only), standard (rotation+translation+intensity), heavy (all)')
    
    # Data split parameters
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of subjects for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of subjects for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Proportion of subjects for testing')
    parser.add_argument('--random_seed', type=lambda x: None if x.lower() == 'none' else int(x), 
                        default=42, help='Random seed for reproducibility (use "none" for no seed)')
    parser.add_argument('--splitting_method', type=str, default='subject', choices=['subject', 'scan'],
                        help='Data splitting method: "subject" (all scans from same subject stay together) or "scan" (individual scans split independently)')
    
    # GPU parameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use')
    
    # Diagnosis filter
    parser.add_argument('--diagnosis_filter', type=str, nargs='+', default=None, help='Filter by diagnosis for TRAIN set (like, CN AD MCI)')
    parser.add_argument('--val_diagnosis_filter', type=str, nargs='+', default=None, help='Filter by diagnosis for VAL set')
    parser.add_argument('--test_diagnosis_filter', type=str, nargs='+', default=None,
                        help='Filter by diagnosis for TEST set (default: same as diagnosis_filter, use "all" for all subjects)')
    
    args = parser.parse_args()
    
    config_file = args.config if args.config else 'config.yaml'
    
    if os.path.exists(config_file):
        print(f"Loading configuration from: {config_file}")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override args with config values
        if 'data' in config:
            if 'data_dir' in config['data']:
                args.data_dir = config['data']['data_dir']
            if 'csv_path' in config['data']:
                args.csv_path = config['data']['csv_path']
            if 'input_shape' in config['data']:
                args.input_shape = config['data']['input_shape']
            if 'train_ratio' in config['data']:
                args.train_ratio = config['data']['train_ratio']
            if 'val_ratio' in config['data']:
                args.val_ratio = config['data']['val_ratio']
            if 'test_ratio' in config['data']:
                args.test_ratio = config['data']['test_ratio']
            if 'random_seed' in config['data']:
                seed_value = config['data']['random_seed']
                # Handle string 'none' or 'null' from YAML
                if isinstance(seed_value, str) and seed_value.lower() in ['none', 'null']:
                    args.random_seed = None
                else:
                    args.random_seed = seed_value
            if 'diagnosis_filter' in config['data']:
                diag_filter = config['data']['diagnosis_filter']
                # Validate diagnosis_filter is a list or None
                if diag_filter is not None and not isinstance(diag_filter, list):
                    raise ValueError(f"diagnosis_filter must be a list or null, got {type(diag_filter).__name__}: {diag_filter}. "
                                   f"Use ['CN'] not 'CN'")
                args.diagnosis_filter = diag_filter
            
            if 'val_diagnosis_filter' in config['data']:
                val_diag_filter = config['data']['val_diagnosis_filter']
                args._val_filter_from_config = True
                if isinstance(val_diag_filter, str) and val_diag_filter.lower() == 'all':
                    args.val_diagnosis_filter = None
                elif val_diag_filter is not None and not isinstance(val_diag_filter, list):
                    raise ValueError(f"val_diagnosis_filter must be a list, 'all', or null")
                else:
                    args.val_diagnosis_filter = val_diag_filter
            else:
                args._val_filter_from_config = False
            
            if 'test_diagnosis_filter' in config['data']:
                test_diag_filter = config['data']['test_diagnosis_filter']
                args._test_filter_from_config = True 
                
                # Handle special "all" keyword
                if isinstance(test_diag_filter, str) and test_diag_filter.lower() == 'all':
                    args.test_diagnosis_filter = None
                elif test_diag_filter is not None and not isinstance(test_diag_filter, list):
                    raise ValueError(f"test_diagnosis_filter must be a list, 'all', or null, got {type(test_diag_filter).__name__}")
                else:
                    args.test_diagnosis_filter = test_diag_filter
            else:
                args._test_filter_from_config = False
            
            # New parameters for original size and slice cropping
            if 'use_original_size' in config['data']:
                args.use_original_size = config['data']['use_original_size']
            if 'crop_slices' in config['data']:
                args.crop_slices = config['data']['crop_slices']
            if 'splitting_method' in config['data']:
                args.splitting_method = config['data']['splitting_method']
        
        if 'model' in config:
            if 'name' in config['model']:
                args.model = config['model']['name']
            if 'config' in config['model']:
                args.model_config = config['model']['config']
            if 'optimizer' in config['model']:
                args.optimizer = config['model']['optimizer']
            if 'use_improved_loss' in config['model']:
                args.use_improved_loss = config['model']['use_improved_loss']
        
        if 'training' in config:
            if 'batch_size' in config['training']:
                args.batch_size = config['training']['batch_size']
            if 'epochs' in config['training']:
                args.epochs = config['training']['epochs']
            if 'iterations' in config['training']:
                args.iterations = config['training']['iterations']
            if 'learning_rate' in config['training']:
                args.learning_rate = config['training']['learning_rate']
            if 'lr_decay' in config['training']:
                args.lr_decay = config['training']['lr_decay']
            if 'patience' in config['training']:
                args.patience = config['training']['patience']
            if 'use_augmentation' in config['training']:
                args.use_augmentation = config['training']['use_augmentation']
            if 'augmentation_config' in config['training']:
                args.augmentation_config = config['training']['augmentation_config']
        
        if 'output' in config:
            if 'output_dir' in config['output']:
                args.output_dir = config['output']['output_dir']

        
        print("Configuration loaded successfully!")
    else:
        print(f"No config file found at '{config_file}', using default settings")
    
    # Validate diagnosis_filter (in case set via command line)
    if args.diagnosis_filter is not None and isinstance(args.diagnosis_filter, str):
        # Convert single string to list:
        args.diagnosis_filter = [args.diagnosis_filter]
        print(f"Note: Converted diagnosis_filter to list: {args.diagnosis_filter}")
    
    # Handle val_diagnosis_filter:
    if not hasattr(args, 'val_diagnosis_filter'):
        args.val_diagnosis_filter = args.diagnosis_filter
        args._val_filter_explicitly_set = False
    elif hasattr(args, '_val_filter_from_config') and args._val_filter_from_config:
        args._val_filter_explicitly_set = True
    elif isinstance(args.val_diagnosis_filter, list) and len(args.val_diagnosis_filter) == 1 and args.val_diagnosis_filter[0].lower() == 'all':
        args.val_diagnosis_filter = None
        args._val_filter_explicitly_set = True
    elif args.val_diagnosis_filter is not None and isinstance(args.val_diagnosis_filter, str):
        args.val_diagnosis_filter = [args.val_diagnosis_filter]
        args._val_filter_explicitly_set = True
    else:
        args._val_filter_explicitly_set = (args.val_diagnosis_filter is None and hasattr(args, '_val_filter_from_config'))
    
    # Set default splitting_method if not provided
    if not hasattr(args, 'splitting_method'):
        args.splitting_method = 'subject'  # Default to subject-level splitting
    
    # Handle test_diagnosis_filter
    # Use a flag to distinguish between "not set" vs "explicitly set to None"
    if not hasattr(args, 'test_diagnosis_filter'):
        # Not set at all - use same as training
        args.test_diagnosis_filter = args.diagnosis_filter
        args._test_filter_explicitly_set = False
    elif hasattr(args, '_test_filter_from_config') and args._test_filter_from_config:
        # Was explicitly set in config (even if to None) - keep it
        args._test_filter_explicitly_set = True
    elif isinstance(args.test_diagnosis_filter, list) and len(args.test_diagnosis_filter) == 1 and args.test_diagnosis_filter[0].lower() == 'all':
        # Handle "all" keyword from command line
        args.test_diagnosis_filter = None
        args._test_filter_explicitly_set = True
    elif args.test_diagnosis_filter is not None and isinstance(args.test_diagnosis_filter, str):
        # Convert single string to list
        args.test_diagnosis_filter = [args.test_diagnosis_filter]
        args._test_filter_explicitly_set = True
    else:
        # Was set to None from command line or config
        args._test_filter_explicitly_set = (args.test_diagnosis_filter is None and hasattr(args, '_test_filter_from_config'))
    
    return args


# PyTorch Model Definitions
class SimpleCNN3D(nn.Module):
    # Simple 3D CNN - used as base model 
    def __init__(self, input_shape):
        super(SimpleCNN3D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )  
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


class ResNet3D(nn.Module):
    #3D ResNet 
    def __init__(self, input_shape):
        super(ResNet3D, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4)
        self.layer3 = self._make_layer(256, 512, 6)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU())
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(-1)


class SFCN_PyTorch(nn.Module):
    # SFCN (Simple Fully Convolutional Network) for brain age prediction
    def __init__(self, input_shape):
        super(SFCN_PyTorch, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


def build_model(model_name, input_shape, device, model_config='standard'):
    """
    Build the specified model architecture in PyTorch.
        # model_name (str): Name of the model
        # input_shape (tuple): Input shape including channel (like, (121, 145, 121, 1))   
        # device: PyTorch device (cpu or cuda)
        # model_config (str): Configuration for our brainagenet_V2 models ('light', 'standard', 'heavy')
    Returns:
        model: PyTorch model
        optimizer_type (str): Recommended optimizer type
    """
    print(f"\nBuilding {model_name} model...")
    
    if model_name == 'densenet':
        model = SimpleCNN3D(input_shape)
        optimizer_type = 'Adam'
    elif model_name == 'resnet':
        model = ResNet3D(input_shape)
        optimizer_type = 'Adam'
    elif model_name == 'vgg16':
        model = SimpleCNN3D(input_shape)
        optimizer_type = 'SGD'
    elif model_name == 'sfcn':
        model = SFCN_PyTorch(input_shape)
        optimizer_type = 'SGD'
    elif model_name == 'brainagenet':
        model = BrainAgeNet(input_shape, dropout_rate=0.4)
        optimizer_type = 'Adam'
    elif model_name == 'brainagenet_v2':
        print(f"  Configuration: {model_config}")
        model = BrainAgeNetV2(input_shape, dropout_rate=0.3, config=model_config)
        optimizer_type = 'Adam'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    print(f"Model {model_name} built successfully and moved to {device}!")
    return model, optimizer_type


def create_optimizer(opt_type, learning_rate, model):
    # Create PyTorch optimizer with specified type and learning rate.
    # Add light weight decay for regularization (very conservative)
    weight_decay = 1e-5  # Very light L2 regularization
    if opt_type == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), 
                        eps=1e-08, weight_decay=weight_decay)
    elif opt_type == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                        weight_decay=weight_decay)
    elif opt_type == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, 
                           eps=1e-08, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def plot_training_history(history, output_path, args=None, timestamp=None):
    # Plot and save training history.
    plt.figure(figsize=(10, 6))
    
    # Plot loss
    plt.plot(history['train_loss'], label='Training Loss', marker='o', markersize=3)
    plt.plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_path, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Log to wandb if args and timestamp are provided
    if args is not None and timestamp is not None:
        try:
            # Check if wandb is already initialized
            if wandb.run is None:
                wandb.init(project="Brain_age_prediction", name=f"{args.model}_{timestamp}", config=vars(args))
            wandb.log({"Training_history": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Warning: Could not log to wandb: {e}")


def evaluate_model(model, dataloader, device, output_path, set_name):
    """
    Evaluate PyTorch model on the dataset.
        # model: Trained PyTorch model
        # dataloader: DataLoader
        # device: (cpu or cuda)
        # output_path: Path to save results
        # set_name: Name of the dataset (train/val/test)
        
    Returns:
        mae: Mean absolute error
        r: Pearson correlation coefficient based on the paper
    """
    print(f"\nEvaluating on {set_name} set...")
    
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc=f"Evaluating {set_name}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            y_true_list.extend(y_batch.cpu().numpy())
            y_pred_list.extend(outputs.cpu().numpy())
    
    y = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate Pearson correlation coefficient (r)
    r = np.corrcoef(y, y_pred)[0, 1]
    
    print(f"{set_name} Results:")
    print(f"  MAE: {mae:.4f} years")
    print(f"  r (correlation): {r:.4f}")
    
    # Save predictions
    results_dict = {
        'y_true': y,
        'y_pred': y_pred,
        'mae': mae,
        'r': r
    }  
    return mae, r

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    # Train for one epoch
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in tqdm(dataloader, desc="Training"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_one_epoch(model, dataloader, criterion, device):
    # Validate for one epoch
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def set_seed(seed):
    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Make CuDNN deterministic for palmetto
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"Random seed set to {seed} for reproducibility")
    else:
        print("No random seed set - results will be non-deterministic")

def train_model(args):
    
    # Set random seeds for reproducibility
    set_seed(args.random_seed)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, 
        f"{args.model}_{timestamp}"
    )
    os.makedirs(output_path, exist_ok=True) 
    
    # Save arguments
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n")
    print("ADNI Brain Age Prediction Training")
    print("\n")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_path}")
    print("\n")
    
    # Initialize dataloader for training (with diagnosis filter)
    print("Initializing dataloader for training...")
    print(f"  Train filter: {args.diagnosis_filter}")
    print(f"  Val filter: {args.val_diagnosis_filter}")
    print(f"  Test filter: {args.test_diagnosis_filter}")
    
    # Handle slice cropping
    crop_slices = None
    if hasattr(args, 'crop_slices') and args.crop_slices is not None:
        crop_slices = tuple(args.crop_slices)
        cropped_depth = crop_slices[1] - crop_slices[0]
        print(f"\nSlice cropping enabled: keeping slices {crop_slices[0]}-{crop_slices[1]-1}")
        print(f"  This will reduce the first dimension to {cropped_depth} slices")
        # If using original size with cropping, update input_shape
        if hasattr(args, 'use_original_size') and args.use_original_size:
            # Original: (256, 256, 170), after cropping: (100, 256, 170)
            args.input_shape = [cropped_depth, args.input_shape[1], args.input_shape[2]]
            print(f"  Updated input_shape to: {args.input_shape}")
    
    # Determine splitting method (from config or command line, default to 'subject')
    splitting_method = getattr(args, 'splitting_method', 'subject')
    print(f"\nData splitting method: {splitting_method}")
    
    # Choose appropriate dataloader based on splitting method
    if splitting_method == 'scan':
        DataLoaderClass = ADNIDataLoaderScanLevel
    else:
        DataLoaderClass = ADNIDataLoader
    
    data_loader_obj = DataLoaderClass(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        target_shape=tuple(args.input_shape),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        diagnosis_filter=args.diagnosis_filter,
        use_original_size=hasattr(args, 'use_original_size') and args.use_original_size,
        crop_slices=crop_slices
    )
    
    # Print dataset statistics
    data_loader_obj.print_statistics()
    
    # Load data
    print("\n")
    print("Loading data...")
    print("\n")
    
    X_train, y_train, meta_train = data_loader_obj.load_data('train', normalize=True)
    
    # Check if we need to load additional subjects for val/test with different filters
    use_separate_val_filter = (
        hasattr(args, '_val_filter_explicitly_set') and 
        args._val_filter_explicitly_set and
        args.val_diagnosis_filter != args.diagnosis_filter
    )
    use_separate_test_filter = (
        hasattr(args, '_test_filter_explicitly_set') and 
        args._test_filter_explicitly_set and
        args.test_diagnosis_filter != args.diagnosis_filter
    )
    
    # If either val or test needs additional subjects, load them ONCE and split properly
    if use_separate_val_filter or use_separate_test_filter:
        print(f"\nLoading additional subjects for validation/test sets...")
        
        # Load the base train/val/test from the training filter
        X_val_base, y_val_base, _ = data_loader_obj.load_data('val', normalize=True)
        X_test_base, y_test_base, _ = data_loader_obj.load_data('test', normalize=True)
        
        # Load all additional subjects (subjects not in the training filter)
        df_all = pd.read_csv(args.csv_path)
        df_all.columns = df_all.columns.str.strip()
        df_all = df_all.dropna(subset=['Subject', 'Age', 'Group', 'Image Data ID'])
        
        # Find subjects that need to be added (union of val and test filters, minus train filter)
        diagnoses_to_add = set()
        if use_separate_val_filter and args.val_diagnosis_filter is None:
            # Val filter is "all" - we need all diagnoses
            diagnoses_to_add = set(df_all['Group'].unique())
        elif use_separate_val_filter and args.val_diagnosis_filter is not None:
            diagnoses_to_add.update(args.val_diagnosis_filter)
        
        if use_separate_test_filter and args.test_diagnosis_filter is None:
            diagnoses_to_add = set(df_all['Group'].unique())
        elif use_separate_test_filter and args.test_diagnosis_filter is not None:
            diagnoses_to_add.update(args.test_diagnosis_filter)
        
        # Remove diagnoses already in training filter
        if args.diagnosis_filter is not None:
            diagnoses_to_add = diagnoses_to_add - set(args.diagnosis_filter)
        
        if len(diagnoses_to_add) > 0:
            print(f"  Additional diagnoses to load: {sorted(diagnoses_to_add)}")
            
            # Filter for additional diagnoses only
            df_additional = df_all[df_all['Group'].isin(diagnoses_to_add)]
            
            # Verify files exist
            valid_rows = []
            for idx, row in df_additional.iterrows():
                subject_id = row['Subject']
                image_id = str(row['Image Data ID']).strip()
                subject_dir = os.path.join(args.data_dir, subject_id)
                image_dir = os.path.join(subject_dir, image_id)
                if os.path.exists(image_dir):
                    nii_files = [f for f in os.listdir(image_dir) if f.endswith('.nii')]
                    if len(nii_files) > 0:
                        row['file_path'] = os.path.join(image_dir, nii_files[0])
                        valid_rows.append(row)
            
            df_additional = pd.DataFrame(valid_rows)
            
            # Get unique additional subjects (not in train filter subjects)
            all_train_subjects = set(data_loader_obj.train_subjects) | set(data_loader_obj.val_subjects) | set(data_loader_obj.test_subjects)
            additional_subjects = [s for s in df_additional['Subject'].unique() if s not in all_train_subjects]
            
            if len(additional_subjects) > 0:
                print(f"  Found {len(additional_subjects)} additional subjects")
                
                # Split additional subjects between val and test using the same ratio
                # Calculate how many should go to val vs test based on original ratios
                val_ratio_of_valtest = args.val_ratio / (args.val_ratio + args.test_ratio)
                
                from sklearn.model_selection import train_test_split
                
                # Get diagnosis groups for stratified split
                subject_groups = df_additional[df_additional['Subject'].isin(additional_subjects)].groupby('Subject')['Group'].first()
                
                additional_val_subjects, additional_test_subjects = train_test_split(
                    additional_subjects,
                    test_size=(1 - val_ratio_of_valtest),
                    random_state=args.random_seed,
                    stratify=subject_groups.values
                )
                
                print(f"  Split: {len(additional_val_subjects)} val subjects, {len(additional_test_subjects)} test subjects")
                
                # Load additional val scans
                if use_separate_val_filter and len(additional_val_subjects) > 0:
                    additional_val_df = df_additional[df_additional['Subject'].isin(additional_val_subjects)]
                    print(f"  Loading {len(additional_val_df)} additional validation scans...")
                    
                    mean_val = np.mean(X_train)
                    std_val = np.std(X_train)
                    
                    X_val_additional = np.zeros((len(additional_val_df), *args.input_shape), dtype=np.float32)
                    y_val_additional = np.zeros(len(additional_val_df), dtype=np.float32)
                    val_successful = 0
                    
                    for idx, (df_idx, row) in enumerate(tqdm(additional_val_df.iterrows(), total=len(additional_val_df), desc="  Loading val scans")):
                        try:
                            img = nib.load(row['file_path'])
                            img_data = img.get_fdata().astype(np.float32)
                            if img_data.shape != tuple(args.input_shape):
                                continue
                            img_data = (img_data - mean_val) / (std_val + 1e-8)
                            X_val_additional[val_successful] = img_data
                            y_val_additional[val_successful] = row['Age']
                            val_successful += 1
                        except Exception:
                            continue
                    
                    if val_successful > 0:
                        X_val_additional = X_val_additional[:val_successful]
                        y_val_additional = y_val_additional[:val_successful]
                        X_val = np.concatenate([X_val_base, X_val_additional], axis=0)
                        y_val = np.concatenate([y_val_base, y_val_additional], axis=0)
                        print(f"  Val set: {X_val_base.shape[0]} base + {val_successful} additional = {X_val.shape[0]} total")
                    else:
                        X_val, y_val = X_val_base, y_val_base
                else:
                    X_val, y_val = X_val_base, y_val_base
                
                # Load additional test scans
                if use_separate_test_filter and len(additional_test_subjects) > 0:
                    additional_test_df = df_additional[df_additional['Subject'].isin(additional_test_subjects)]
                    print(f"  Loading {len(additional_test_df)} additional test scans...")
                    
                    mean_val = np.mean(X_train)
                    std_val = np.std(X_train)
                    
                    X_test_additional = np.zeros((len(additional_test_df), *args.input_shape), dtype=np.float32)
                    y_test_additional = np.zeros(len(additional_test_df), dtype=np.float32)
                    test_successful = 0
                    
                    for idx, (df_idx, row) in enumerate(tqdm(additional_test_df.iterrows(), total=len(additional_test_df), desc="  Loading test scans")):
                        try:
                            img = nib.load(row['file_path'])
                            img_data = img.get_fdata().astype(np.float32)
                            if img_data.shape != tuple(args.input_shape):
                                continue
                            img_data = (img_data - mean_val) / (std_val + 1e-8)
                            X_test_additional[test_successful] = img_data
                            y_test_additional[test_successful] = row['Age']
                            test_successful += 1
                        except Exception:
                            continue
                    
                    if test_successful > 0:
                        X_test_additional = X_test_additional[:test_successful]
                        y_test_additional = y_test_additional[:test_successful]
                        X_test = np.concatenate([X_test_base, X_test_additional], axis=0)
                        y_test = np.concatenate([y_test_base, y_test_additional], axis=0)
                        print(f"  Test set: {X_test_base.shape[0]} base + {test_successful} additional = {X_test.shape[0]} total")
                    else:
                        X_test, y_test = X_test_base, y_test_base
                else:
                    X_test, y_test = X_test_base, y_test_base
            else:
                # No additional subjects found
                X_val, y_val = X_val_base, y_val_base
                X_test, y_test = X_test_base, y_test_base
        else:
            # No additional diagnoses to add
            X_val, y_val = X_val_base, y_val_base
            X_test, y_test = X_test_base, y_test_base
    else:
        # Use simple loading (no additional subjects needed)
        X_val, y_val, meta_val = data_loader_obj.load_data('val', normalize=True)
        X_test, y_test, meta_test = data_loader_obj.load_data('test', normalize=True)
    
    # Reshape for PyTorch: (N, C, D, H, W) format
    # Original shape: (N, D, H, W) -> Add channel dimension
    X_train = X_train.reshape(X_train.shape[0], 1, *args.input_shape)
    X_val = X_val.reshape(X_val.shape[0], 1, *args.input_shape)
    X_test = X_test.reshape(X_test.shape[0], 1, *args.input_shape)
    
    print("\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Save data split information
    split_info = {
        'train_filter': args.diagnosis_filter,
        'val_filter': args.val_diagnosis_filter,
        'test_filter': args.test_diagnosis_filter,
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'random_seed': args.random_seed
    }
    with open(os.path.join(output_path, 'data_split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    if args.val_diagnosis_filter != args.diagnosis_filter or args.test_diagnosis_filter != args.diagnosis_filter:
        print(f"\nNote: Using different diagnosis filters!")
        print(f"  Train: {args.diagnosis_filter}")
        print(f"  Val:   {args.val_diagnosis_filter}")
        print(f"  Test:  {args.test_diagnosis_filter}")
    
    # Convert to PyTorch tensors and create DataLoaders with optional augmentation
    # Reshape to (N, D, H, W) format - AugmentedTensorDataset will handle adding channel dimension
    X_train_reshaped = X_train.reshape(X_train.shape[0], *args.input_shape)
    X_val_reshaped = X_val.reshape(X_val.shape[0], *args.input_shape)
    X_test_reshaped = X_test.reshape(X_test.shape[0], *args.input_shape)
    
    # Setup augmentation if enabled
    augmentation = None
    if hasattr(args, 'use_augmentation') and args.use_augmentation:
        aug_config = args.augmentation_config if hasattr(args, 'augmentation_config') else 'standard'
        augmentation = get_augmentation(aug_config)
        print(f"\nData augmentation enabled: {aug_config} configuration")
        print(f"  - Rotation: ±10 degrees")
        if aug_config in ['standard', 'heavy']:
            print(f"  - Translation: enabled")
            print(f"  - Scaling: enabled")
            print(f"  - Intensity adjustments: enabled")
        if aug_config == 'heavy':
            print(f"  - Elastic deformation: enabled")
    else:
        print("\nData augmentation disabled")
    
    # Create datasets with augmentation (only for training)
    train_dataset = AugmentedTensorDataset(
        X_train_reshaped, y_train, mode='train', augmentation=augmentation
    )
    val_dataset = AugmentedTensorDataset(
        X_val_reshaped, y_val, mode='val', augmentation=None
    )
    test_dataset = AugmentedTensorDataset(
        X_test_reshaped, y_test, mode='test', augmentation=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Build model
    # input_shape is already adjusted above if cropping is used
    input_shape = (*args.input_shape, 1)
    model_config = args.model_config if hasattr(args, 'model_config') else 'standard'
    model, optimizer_type = build_model(args.model, input_shape, device, model_config)
    
    # Override optimizer if specified
    if args.optimizer:
        optimizer_type = args.optimizer
    
    # Print model summary
    print("\n")
    print("Model Architecture")
    print("\n")
    print(model)
    print("\n")
    
    # Save model architecture
    with open(os.path.join(output_path, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
        f.write(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Loss function
    if hasattr(args, 'use_improved_loss') and args.use_improved_loss:
        # Use lower alpha (0.3) for more conservative correlation weighting
        criterion = ImprovedLoss(alpha=0.3)
        print("Using improved loss (MAE + correlation, alpha=0.3)")
    else:
        criterion = nn.L1Loss()  # MAE loss
        print("Using standard MAE loss")
    
    # Training loop
    print("\n")
    print("Starting training...")
    print("\n")
    
    learning_rate = args.learning_rate
    best_val_mae = float('inf')
    training_history = {'train_loss': [], 'val_loss': []}
    
    for iteration in range(args.iterations):
        print("\n")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"Learning rate: {learning_rate:.6f}")
        print("\n")
        
        # Load weights from previous iteration if not first iteration
        if iteration > 0:
            weights_path = os.path.join(output_path, 'best_weights.pth')
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=device))
                print(f"Loaded weights from previous iteration")
            learning_rate *= args.lr_decay
            print(f"Reduced learning rate to {learning_rate:.6f}")
        
        # Create optimizer
        optimizer = create_optimizer(optimizer_type, learning_rate, model)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        
        # Early stopping variables
        patience = getattr(args, 'patience', 10)  # Get from args, default to 10
        patience_counter = 0
        best_iter_val_loss = float('inf')
        
        # Training for this iteration
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Log metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model for this iteration
            if val_loss < best_iter_val_loss:
                best_iter_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(output_path, 'best_weights.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best weights from this iteration
        model.load_state_dict(torch.load(os.path.join(output_path, 'best_weights.pth'), map_location=device))
        
        # Evaluate on validation set
        val_mae, val_r = evaluate_model(model, val_loader, device, output_path, f'val_iter{iteration+1}')
        
        # Check if this is the best model overall
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f"\nNew best validation MAE: {best_val_mae:.4f}")
            # Save as final best model
            torch.save(model.state_dict(), os.path.join(output_path, 'final_best_weights.pth'))
    
    # Save training history
    with open(os.path.join(output_path, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Plot training history
    plot_training_history(training_history, output_path, args, timestamp)
    
    # Final evaluation on all sets with best model
    print("\n")
    print("Final Evaluation with Best Model")
    print("\n")
    
    model.load_state_dict(torch.load(os.path.join(output_path, 'final_best_weights.pth'), map_location=device))
    
    train_mae, train_r = evaluate_model(model, train_loader, device, output_path, 'train_final')
    val_mae, val_r = evaluate_model(model, val_loader, device, output_path, 'val_final')
    test_mae, test_r = evaluate_model(model, test_loader, device, output_path, 'test_final')
    
    # Save final results
    final_results = {
        'train_mae': float(train_mae),
        'train_r': float(train_r),
        'val_mae': float(val_mae),
        'val_r': float(val_r),
        'test_mae': float(test_mae),
        'test_r': float(test_r),
        'best_val_mae': float(best_val_mae),
        'model': args.model,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_path, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n")
    print("Training Complete!")
    print("\n")
    print(f"Final Results:")
    print(f"  Train MAE: {train_mae:.4f} years  |  r: {train_r:.4f}")
    print(f"  Val MAE:   {val_mae:.4f} years  |  r: {val_r:.4f}")
    print(f"  Test MAE:  {test_mae:.4f} years  |  r: {test_r:.4f}")
    print(f"\nResults saved to: {output_path}")
    print("\n")
    
    return model, final_results


def main():
    args = parse_args()
    model, results = train_model(args)
    return model, results


if __name__ == '__main__':
    main()

