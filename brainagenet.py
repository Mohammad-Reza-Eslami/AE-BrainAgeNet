"""
BrainAgeNet: Custom Attention-Enhanced 3D CNN for Brain Age Prediction

Architecture Features:
- Squeeze-and-Excitation (SE) blocks for channel-wise attention
- Residual connections for stable gradient flow
- Progressive feature extraction optimized for limited data
- Multi-scale feature aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D convolutions - Channel attention mechanism
    
    Reference:
        Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for bottleneck (default: 8)
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock3D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze: Global spatial pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: Learn channel-wise attention
        y = self.excitation(y).view(b, c, 1, 1, 1)
        # Scale: Apply attention weights
        return x * y.expand_as(x)


class ResidualSEBlock3D(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention
    
    Combines residual learning with channel attention for better feature learning.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for downsampling (default: 1)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualSEBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # SE attention
        self.se = SEBlock3D(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention
        out = self.se(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out


class BrainAgeNet(nn.Module):
    """
    Custom attention-enhanced 3D CNN for brain age prediction.
    
    Key Features:
    - Squeeze-and-Excitation (SE) blocks for channel-wise attention
    - Residual connections for stable gradient flow
    - Progressive feature extraction optimized for limited data
    - Multi-scale feature aggregation
    - Efficient parameter usage to prevent overfitting
    
    Architecture:
        Input (1, D, H, W)
            ↓
        Initial Conv (7×7×7, stride=2) + MaxPool
            ↓ [32 channels]
        Residual SE Block × 2
            ↓ [64 channels]
        Residual SE Block × 2 (stride=2)
            ↓ [128 channels]
        Residual SE Block × 2 (stride=2)
            ↓ [256 channels]
        Residual SE Block × 2 (stride=2)
            ↓ [512 channels]
        Global Average Pooling
            ↓
        FC (512→256) + Dropout + ReLU
            ↓
        FC (256→128) + Dropout + ReLU
            ↓
        FC (128→1)
            ↓
        Predicted Age
    
    Args:
        input_shape (tuple): Shape of input MRI volume (D, H, W)
        dropout_rate (float): Dropout rate for regularization (default: 0.4)
    
    Example:
        model = BrainAgeNet(input_shape=(121, 145, 121))
        x = torch.randn(4, 1, 121, 145, 121)
        age_pred = model(x)
        print(age_pred.shape)  # torch.Size([4])
    """
    def __init__(self, input_shape, dropout_rate=0.4):
        super(BrainAgeNet, self).__init__()
        
        # Initial convolution - Extract low-level features
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual SE blocks - Progressive feature learning
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Global pooling - Aggregate spatial information
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        # Regression head with dropout for regularization
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        # Create a layer with multiple residual SE blocks
        layers = []
        # First block with stride for downsampling
        layers.append(ResidualSEBlock3D(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualSEBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        # Initialize network weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass  
        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, D, H, W)
        Returns:
            torch.Tensor: Predicted ages of shape (N,)
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and regression
        x = self.avgpool(x)
        x = self.fc(x)
        
        return x.squeeze(-1)


def get_model(input_shape=(121, 145, 121), dropout_rate=0.4):
    """
    Create a function to create BrainAgeNet model
    
    Args:
        input_shape (tuple): Shape of input MRI volume
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        BrainAgeNet: Instantiated model
    """
    return BrainAgeNet(input_shape, dropout_rate)


if __name__ == "__main__":
    # Quick test
    print("Testing BrainAgeNet...")
    model = BrainAgeNet(input_shape=(121, 145, 121))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 121, 145, 121)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Model works!")

