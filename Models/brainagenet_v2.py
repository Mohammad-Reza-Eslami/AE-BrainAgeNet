# BrainAgeNet V2: Enhanced Attention-Enhanced 3D CNN for Brain Age Prediction
"""
- Enhanced Channel Attention: SE block with both avg and max pooling
- Optional Spatial Attention: Added only in later layers (standard config) to avoid overfitting
- Full CBAM: Available in heavy config but disabled by default due to complexity
- Improved loss function (MAE + Correlation Loss)

Reference:
    - Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    - Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention3D(nn.Module):
    # Channel Attention Module for 3D convolutions (Enhanced SE Block)
    # Uses both average and max pooling for richer feature representation.
    def __init__(self, channels, reduction=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1, 1)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1, 1)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    # Spatial Attention Module for 3D convolutions
    # Focuses on 'where' informative features are located spatially.
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics: avg and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        x_attention = torch.cat([avg_out, max_out], dim=1)
        x_attention = self.conv(x_attention)
        
        return x * self.sigmoid(x_attention)


class CBAM3D(nn.Module):
    # Convolutional Block Attention Module (CBAM) for 3D
    # Combines Channel Attention and Spatial Attention sequentially.
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size=spatial_kernel)
    
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x


class SelfAttention3D(nn.Module):
    # Self-Attention module for 3D feature maps
    # Captures long-range dependencies in the feature space.
    def __init__(self, channels, reduction=4):
        super(SelfAttention3D, self).__init__()
        self.channels = channels
        self.reduced_channels = channels // reduction
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv3d(channels, self.reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(channels, self.reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        b, c, d, h, w = x.size()
        
        # Flatten spatial dimensions
        proj_query = self.query_conv(x).view(b, self.reduced_channels, -1)  # (B, C', N)
        proj_key = self.key_conv(x).view(b, self.reduced_channels, -1)      # (B, C', N)
        proj_value = self.value_conv(x).view(b, c, -1)                      # (B, C, N)
        
        # Transpose for matrix multiplication
        proj_query = proj_query.permute(0, 2, 1)  # (B, N, C')
        proj_key = proj_key.permute(0, 2, 1)      # (B, N, C')
        proj_value = proj_value.permute(0, 2, 1)   # (B, N, C)
        
        # Compute attention
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))  # (B, N, N)
        attention = self.softmax(energy)
        
        # Apply attention to values
        out = torch.bmm(attention, proj_value)  # (B, N, C)
        out = out.permute(0, 2, 1).view(b, c, d, h, w)  # (B, C, D, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class ResidualCBAMBlock3D(nn.Module):
    # Residual block with CBAM (Channel + Spatial Attention)  
    # Enhanced version of ResidualSEBlock with dual attention.

    def __init__(self, in_channels, out_channels, stride=1, use_self_attention=False):
        super(ResidualCBAMBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # CBAM attention (Channel + Spatial)
        self.cbam = CBAM3D(out_channels)
        
        # Optional self-attention for long-range dependencies
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_attn = SelfAttention3D(out_channels)
        
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
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        # Apply self-attention if enabled
        if self.use_self_attention:
            out = self.self_attn(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out


class ResidualEnhancedSEBlock3D(nn.Module):
    # Residual block with Enhanced SE (Channel Attention with avg+max pooling)
    # Lighter than CBAM - only channel attention, but enhanced with max pooling.
    def __init__(self, in_channels, out_channels, stride=1, use_spatial_attention=False):
        super(ResidualEnhancedSEBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Enhanced channel attention (avg + max pooling)
        self.channel_attention = ChannelAttention3D(out_channels)
        
        # Optional spatial attention (only in later layers)
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention3D(kernel_size=7)
        
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
        
        # Apply enhanced channel attention
        out = self.channel_attention(out)
        
        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            out = self.spatial_attention(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out


class MultiScaleFeatureFusion(nn.Module):
    # Multi-scale feature fusion with attention weighting
    # Aggregates features from different scales with learned attention weights.
    def __init__(self, channels_list):
        super(MultiScaleFeatureFusion, self).__init__()
        self.num_scales = len(channels_list)
        
        # Attention weights for each scale
        self.attention_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Projection layers to align channel dimensions
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(ch, channels_list[-1], kernel_size=1),
                nn.BatchNorm3d(channels_list[-1])
            ) for ch in channels_list[:-1]
        ])
    
    def forward(self, features_list):
        # Normalize attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        
        # Project all features to the same channel dimension
        projected_features = []
        for i, feat in enumerate(features_list[:-1]):
            projected = self.projections[i](feat)
            # Upsample to match spatial dimensions of the largest feature
            if feat.shape[2:] != features_list[-1].shape[2:]:
                projected = F.interpolate(
                    projected, 
                    size=features_list[-1].shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
            projected_features.append(projected)
        
        # Add the largest feature (already correct size)
        projected_features.append(features_list[-1])
        
        # Weighted fusion
        fused = sum(w * feat for w, feat in zip(weights, projected_features))
        
        return fused


class BrainAgeNetV2(nn.Module):

    def __init__(self, input_shape, dropout_rate=0.4, config='standard'):
        super(BrainAgeNetV2, self).__init__()
        
        self.config = config
        
        # Configuration-dependent settings
        # Ultra-conservative approach
        # Enhanced channel attention (avg+max pooling)
        if config == 'light':
            num_blocks = [2, 2, 2, 2]  
            use_spatial_attention = False
            use_feature_refinement = False
        elif config == 'standard':
            num_blocks = [2, 2, 2, 2]  
            use_spatial_attention = False
            use_feature_refinement = False  
        elif config == 'heavy':
            num_blocks = [2, 2, 2, 2]  
            use_spatial_attention = True  # Only in heavy, only in last layer
            use_feature_refinement = False 
        else:
            raise ValueError(f"Unknown config: {config}. Must be 'light', 'standard', or 'heavy'")
        
        self.use_spatial_attention = use_spatial_attention
        self.use_feature_refinement = use_feature_refinement
        
        # Initial convolution - Extract low-level features
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks - Progressive feature learning
        # Enhanced SE (channel attention with avg+max pooling)
        # Only add spatial attention in heavy config, and only in last layer
        self.layer1 = self._make_enhanced_se_layer(32, 64, num_blocks[0], stride=1, use_spatial=False)
        self.layer2 = self._make_enhanced_se_layer(64, 128, num_blocks[1], stride=2, use_spatial=False)
        self.layer3 = self._make_enhanced_se_layer(128, 256, num_blocks[2], stride=2, use_spatial=False)
        self.layer4 = self._make_enhanced_se_layer(256, 512, num_blocks[3], stride=2, use_spatial=use_spatial_attention)  # Only here if heavy
        
        # Global pooling - Aggregate spatial information
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Back to original dropout
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_self_attention=False):
        # Create a layer with multiple residual CBAM blocks
        layers = []
        # First block with stride for downsampling
        layers.append(ResidualCBAMBlock3D(in_channels, out_channels, stride, use_self_attention=False))
        # Remaining blocks (self-attention only in middle blocks if enabled)
        for i in range(1, num_blocks):
            use_sa = use_self_attention and (i == num_blocks - 1)  # Only last block
            layers.append(ResidualCBAMBlock3D(out_channels, out_channels, stride=1, use_self_attention=use_sa))
        return nn.Sequential(*layers)
    
    def _make_enhanced_se_layer(self, in_channels, out_channels, num_blocks, stride, use_spatial=False):
        # Create a layer with Enhanced SE blocks (lighter than CBAM)
        layers = []
        layers.append(ResidualEnhancedSEBlock3D(in_channels, out_channels, stride, use_spatial_attention=use_spatial))
        for _ in range(1, num_blocks):
            layers.append(ResidualEnhancedSEBlock3D(out_channels, out_channels, stride=1, use_spatial_attention=use_spatial))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        # Initialize network weights
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


class ImprovedLoss(nn.Module):
    # Improved loss function combining MAE with correlation loss
    #This loss encourages the model to minimize absolute error + maintain high correlation between predictions and ground truth.
    # Loss = MAE + α * (1 - Correlation)
    # where α is a weighting factor (default: 0.3, lower is more conservative)
    def __init__(self, alpha=0.3, lambda_corr=None):
        super(ImprovedLoss, self).__init__()
        # Support both 'alpha' and 'lambda_corr' for compatibility
        if lambda_corr is not None:
            self.alpha = lambda_corr
        else:
            self.alpha = alpha
        self.mae_loss = nn.L1Loss()
    
    def forward(self, predictions, targets):

        # MAE loss
        mae = self.mae_loss(predictions, targets)
        
        # Correlation loss
        # Compute Pearson correlation coefficient
        pred_mean = predictions.mean()
        target_mean = targets.mean()
        
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        
        correlation = numerator / (pred_std * target_std + 1e-8)
        
        # Correlation loss: (1 - correlation) to maximize correlation
        corr_loss = 1 - correlation
        
        # Combined loss
        total_loss = mae + self.alpha * corr_loss
        
        return total_loss


def get_model(input_shape=(121, 145, 121), dropout_rate=0.4, config='standard'):

    return BrainAgeNetV2(input_shape, dropout_rate, config)



