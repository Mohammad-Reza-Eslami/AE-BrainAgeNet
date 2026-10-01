# Training script for ADNI Brain Age Prediction using 3D CNNs

# With this code we can train various deep learning models (DenseNet, ResNet, VGG16, SFCN) from the 2022 paper
# on the ADNI dataset for brain age prediction.
import os
import sys
import numpy as np
import argparse
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import wandb

# Keras/TensorFlow imports
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
import scipy.io as sio

# Import models from the model folder
sys.path.append('model')
from densenet3d_regression import build_densenet_forCAM  # type: ignore
from resnet_3d import Resnet3DBuilder  # type: ignore
from vgg_16 import vgg16_3D  # type: ignore
from sfcn_keras import SFCN  # type: ignore

# Import dataloader
from adni_dataloader import ADNIDataLoader


class TrainingMonitor(Callback):
    # Custom callback to monitor training progress.
    def __init__(self, output_path):
        super().__init__()
        self.output_path = output_path
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        
        # Save history
        with open(os.path.join(self.output_path, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Train Brain Age Prediction Model on ADNI Dataset')
    
    # Config file option
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides other arguments)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Datasets/ADNI_Merged',
                        help='Path to ADNI_Merged directory')
    parser.add_argument('--csv_path', type=str, default='Datasets/ADNI1_Combined_Unique_ImageDataID.csv',
                        help='Path to CSV file with metadata')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='densenet', 
                        choices=['densenet', 'resnet', 'vgg16', 'sfcn'],
                        help='Model architecture to use')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[121, 145, 121],
                        help='Input shape for MRI images')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs per iteration')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='Learning rate decay factor per iteration')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD', 'RMSprop'],
                        help='Optimizer to use')
    
    # Data split parameters
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion of subjects for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion of subjects for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proportion of subjects for testing')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU parameters
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID to use')
    
    # Diagnosis filter
    parser.add_argument('--diagnosis_filter', type=str, nargs='+', default=None,
                        help='Filter by diagnosis (e.g., CN AD MCI)')
    
    args = parser.parse_args()
    
    # Auto-load config.yaml if it exists (even if not specified)
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
                args.random_seed = config['data']['random_seed']
            if 'diagnosis_filter' in config['data']:
                args.diagnosis_filter = config['data']['diagnosis_filter']
        
        if 'model' in config:
            if 'name' in config['model']:
                args.model = config['model']['name']
            if 'optimizer' in config['model']:
                args.optimizer = config['model']['optimizer']
        
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
        
        if 'output' in config:
            if 'output_dir' in config['output']:
                args.output_dir = config['output']['output_dir']

        
        print("Configuration loaded successfully!")
    else:
        print(f"No config file found at '{config_file}', using default settings")
    
    return args


def build_model(model_name, input_shape):
    """
    Build the specified model architecture.
        # model_name (str): Name of the model ('densenet', 'resnet', 'vgg16', 'sfcn')
        # input_shape (tuple): Input shape including channel (like, (121, 145, 121, 1))   
    Returns:
        model: Keras model
        optimizer_type (str): Recommended optimizer type
    """
    print(f"\nBuilding {model_name} model...")
    
    if model_name == 'densenet':
        model = build_densenet_forCAM(input_shape, 0)
        optimizer_type = 'Adam'
    elif model_name == 'resnet':
        model = Resnet3DBuilder.build_resnet_101(input_shape, 1)
        optimizer_type = 'Adam'
    elif model_name == 'vgg16':
        model = vgg16_3D(input_shape, 1)
        optimizer_type = 'SGD'
    elif model_name == 'sfcn':
        model = SFCN(input_shape, 'False')
        optimizer_type = 'SGD'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Model {model_name} built successfully!")
    return model, optimizer_type


def create_optimizer(opt_type, learning_rate):
    # Create optimizer with specified type and learning rate.
    # Keras 3.x uses 'learning_rate' instead of 'lr'
    if opt_type == 'Adam':
        return Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif opt_type == 'SGD':
        return SGD(learning_rate=learning_rate)
    elif opt_type == 'RMSprop':
        return RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def plot_training_history(history, output_path):
    # Plot and save training history.
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_history.png'), dpi=150)
    plt.close()


def evaluate_model(model, X, y, batch_size, output_path, set_name):
    """
    Evaluate model on a dataset.
    
        # model: Trained Keras model
        # X: Input data
        # y: True ages
        # batch_size: Batch size for prediction
        # output_path: Path to save results
        # set_name: Name of the dataset (train/val/test)
        
    Returns:
        mae: Mean absolute error
        r: Pearson correlation coefficient
    """
    print(f"\nEvaluating on {set_name} set...")
    
    # Make predictions
    y_pred = model.predict(X, batch_size=batch_size, verbose=1)
    y_pred = y_pred.flatten()
    
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
    
    # Also save as numpy
    np.savez(os.path.join(output_path, f'{set_name}_predictions.npz'), **results_dict)
    
    # Create scatter plot
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.title(f'{set_name}: True vs Predicted Age\nMAE={mae:.2f}, r={r:.3f}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{set_name}_scatter.png'), dpi=150)
    
    # Log to wandb
    wandb.log({f'{set_name}_scatter': wandb.Image(fig), f'{set_name}_mae': mae, f'{set_name}_r': r})
    
    plt.close()
    
    return mae, r


def train_model(args):
    
    # Set GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, 
        f"{args.model}_{timestamp}"
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="Brain_age_prediction", name=f"{args.model}_{timestamp}", config=vars(args))
    
    # Save arguments
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n")
    print("ADNI Brain Age Prediction Training")
    print("\n")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_path}")
    
    # Initialize dataloader
    print("Initializing dataloader...")
    dataloader = ADNIDataLoader(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        target_shape=tuple(args.input_shape),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        diagnosis_filter=args.diagnosis_filter
    )
    
    # Print dataset statistics
    dataloader.print_statistics()
    
    # Load data
    print("\n")
    print("Loading data...")
    print("\n")
    
    X_train, y_train, meta_train = dataloader.load_data('train', normalize=True)
    X_val, y_val, meta_val = dataloader.load_data('val', normalize=True)
    X_test, y_test, meta_test = dataloader.load_data('test', normalize=True)
    
    # Reshape to add channel dimension
    X_train = X_train.reshape(X_train.shape[0], *args.input_shape, 1)
    X_val = X_val.reshape(X_val.shape[0], *args.input_shape, 1)
    X_test = X_test.reshape(X_test.shape[0], *args.input_shape, 1)
    
    print("\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Build model
    input_shape = (*args.input_shape, 1)
    model, optimizer_type = build_model(args.model, input_shape)
    
    # Override optimizer if specified
    if args.optimizer:
        optimizer_type = args.optimizer
    
    # Print model summary
    model.summary()
    
    # Save model architecture
    try:
        plot_model(model, to_file=os.path.join(output_path, 'model_architecture.png'), 
                   show_shapes=True)
    except:
        print("Could not save model architecture plot")
    
    with open(os.path.join(output_path, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Training loop
    print("\n")
    print("Starting training...")
    print("\n")
    
    learning_rate = args.learning_rate
    best_val_mae = float('inf')
    
    for iteration in range(args.iterations):
        print("\n")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"Learning rate: {learning_rate:.6f}")
        print("\n")
        
        # Load weights from previous iteration if not first iteration
        if iteration > 0:
            weights_path = os.path.join(output_path, 'best_weights.h5')
            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                print(f"Loaded weights from previous iteration")
            learning_rate *= args.lr_decay
            print(f"Reduced learning rate to {learning_rate:.6f}")
        
        # Create optimizer
        optimizer = create_optimizer(optimizer_type, learning_rate)
        
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer=optimizer)
        
        # Setup callbacks
        checkpoint_path = os.path.join(output_path, 'best_weights.weights.h5')
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_loss', 
                           save_best_only=True, save_weights_only=True, mode='min', verbose=1),
            EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min'),
            CSVLogger(os.path.join(output_path, f'training_log_iter{iteration+1}.csv')),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                            min_lr=1e-7, verbose=1)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            shuffle=True,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        # Load best weights
        model.load_weights(checkpoint_path)
        
        # Evaluate on validation set
        val_mae, val_r = evaluate_model(model, X_val, y_val, args.batch_size, 
                                        output_path, f'val_iter{iteration+1}')
        
        # Check if this is the best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f"\nNew best validation MAE: {best_val_mae:.4f}")
            # Save as final best model
            model.save_weights(os.path.join(output_path, 'final_best_weights.weights.h5'))
        
        # Save iteration results
        with open(os.path.join(output_path, f'iteration_{iteration+1}_results.txt'), 'w') as f:
            f.write(f"Iteration: {iteration + 1}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Validation MAE: {val_mae:.4f}\n")
            f.write(f"Validation r: {val_r:.4f}\n")
    
    # Final evaluation on all sets with best model
    print("\n")
    print("Final Evaluation with Best Model")
    print("\n")
    
    model.load_weights(os.path.join(output_path, 'final_best_weights.weights.h5'))
    
    train_mae, train_r = evaluate_model(model, X_train, y_train, args.batch_size, 
                                        output_path, 'train_final')
    val_mae, val_r = evaluate_model(model, X_val, y_val, args.batch_size, 
                                    output_path, 'val_final')
    test_mae, test_r = evaluate_model(model, X_test, y_test, args.batch_size, 
                                      output_path, 'test_final')
    
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
    
    # Finish wandb
    wandb.finish()
    
    return model, final_results


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    model, results = train_model(args)
    return model, results


if __name__ == '__main__':
    main()

