#GAN.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class UNetBlock(nn.Module):
    """Basic U-Net building block with skip connections"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        self.down = down
        
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            
        self.norm = nn.InstanceNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.dropout:
            x = self.dropout(x)
        return F.leaky_relu(x, 0.2) if self.down else F.relu(x)

class Generator(nn.Module):
    """
    U-Net Generator with noise injection and multi-channel output Maintains the same interface as the original Generator
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- ENCODER (Downsampling path) ---
        # Input: (B, 4, 88, 128)
        self.down1 = nn.Conv2d(config.CHANNELS, 64, 4, 2, 1)  # -> (B, 64, 44, 64)
        self.down2 = UNetBlock(64, 128, down=True)             # -> (B, 128, 22, 32)
        self.down3 = UNetBlock(128, 256, down=True)            # -> (B, 256, 11, 16)
        self.down4 = UNetBlock(256, 512, down=True)            # -> (B, 512, 5, 8)
        self.down5 = UNetBlock(512, 512, down=True)            # -> (B, 512, 2, 4)
        self.down6 = UNetBlock(512, 512, down=True)            # -> (B, 512, 1, 2)
        
        # --- BOTTLENECK with Noise Injection ---
        # Calculate the bottleneck feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.CHANNELS, config.PITCHES, config.TIMESTEPS)
            x = F.leaky_relu(self.down1(dummy_input), 0.2)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            bottleneck_out = self.down6(x)
            self.bottleneck_size = bottleneck_out.shape[1] * bottleneck_out.shape[2] * bottleneck_out.shape[3]
        
        # Noise projection layer to match bottleneck dimensions
        self.noise_projection = nn.Linear(config.NOISE_DIM, self.bottleneck_size)
        
        # Bottleneck processing
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )
        
        # --- DECODER (Upsampling path with skip connections) ---
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)     # -> (B, 512, 2, 4)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)    # -> (B, 512, ?, ?) 
        self.up3 = UNetBlock(1024, 256, down=False, use_dropout=True)    # -> (B, 256, ?, ?)
        self.up4 = UNetBlock(512, 128, down=False)                       # -> (B, 128, ?, ?)
        self.up5 = UNetBlock(256, 64, down=False)                        # -> (B, 64, ?, ?)
        
        # Final output layer
        self.final = nn.ConvTranspose2d(128, config.CHANNELS, 4, 2, 1)   # -> (B, 4, 88, 128)
        
    def forward(self, content_roll, noise):
        # --- ENCODER (with skip connection storage) ---
        # content_roll shape: (B, 4, 88, 128)
        e1 = F.leaky_relu(self.down1(content_roll), 0.2)  # (B, 64, 44, 64)
        e2 = self.down2(e1)                               # (B, 128, 22, 32)
        e3 = self.down3(e2)                               # (B, 256, 11, 16)
        e4 = self.down4(e3)                               # (B, 512, 5, 8)
        e5 = self.down5(e4)                               # (B, 512, 2, 4)
        e6 = self.down6(e5)                               # (B, 512, 1, 2)
        
        # --- NOISE INJECTION at bottleneck ---
        batch_size = content_roll.size(0)
        
        # Project noise to match bottleneck spatial dimensions
        noise_projected = self.noise_projection(noise)  # (B, bottleneck_size)
        noise_reshaped = noise_projected.view(batch_size, e6.shape[1], e6.shape[2], e6.shape[3])
        
        # Combine content features with noise
        bottleneck_input = e6 + noise_reshaped
        bottleneck_out = self.bottleneck(bottleneck_input)
        
        # --- DECODER (with skip connections and size matching) ---
        d1 = self.up1(bottleneck_out)                     # (B, 512, 2, 4)
        
        # Match sizes for skip connection using interpolation if needed
        if d1.shape[2:] != e5.shape[2:]:
            d1 = F.interpolate(d1, size=e5.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e5], dim=1)                   # (B, 1024, 2, 4)
        
        d2 = self.up2(d1)                                 # (B, 512, ?, ?)
        if d2.shape[2:] != e4.shape[2:]:
            d2 = F.interpolate(d2, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e4], dim=1)                   # (B, 1024, 5, 8)
        
        d3 = self.up3(d2)                                 # (B, 256, ?, ?)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)                   # (B, 512, 11, 16)
        
        d4 = self.up4(d3)                                 # (B, 128, ?, ?)
        if d4.shape[2:] != e2.shape[2:]:
            d4 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e2], dim=1)                   # (B, 256, 22, 32)
        
        d5 = self.up5(d4)                                 # (B, 64, ?, ?)
        if d5.shape[2:] != e1.shape[2:]:
            d5 = F.interpolate(d5, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e1], dim=1)                   # (B, 128, 44, 64)
        
        # --- FINAL OUTPUT ---
        raw_output = self.final(d5)                       # (B, 4, 88, 128)
        
        # Apply unified tanh activation 
        final_output = torch.sigmoid(raw_output)
        
        return final_output

class Discriminator(nn.Module):
    """
    U-Net style Discriminator for music classification Maintains the same interface as the original Discriminator
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- ENCODER PATH ---
        # Input: (B, 4, 88, 128)
        self.down1 = nn.Conv2d(config.CHANNELS, 64, 4, 2, 1)    # -> (B, 64, 44, 64)
        self.down2 = UNetBlock(64, 128, down=True)               # -> (B, 128, 22, 32)
        self.down3 = UNetBlock(128, 256, down=True)              # -> (B, 256, 11, 16)
        self.down4 = UNetBlock(256, 512, down=True)              # -> (B, 512, 5, 8)
        self.down5 = UNetBlock(512, 512, down=True)              # -> (B, 512, 2, 4)
        
        # --- FEATURE PROCESSING ---
        # Global average pooling for temporal aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        # --- OPTIONAL: Add spatial attention mechanism ---
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # --- FEATURE EXTRACTION ---
        # x shape: (B, 4, 88, 128)
        x = F.leaky_relu(self.down1(x), 0.2)  # (B, 64, 44, 64)
        x = self.down2(x)                     # (B, 128, 22, 32)
        x = self.down3(x)                     # (B, 256, 11, 16)
        x = self.down4(x)                     # (B, 512, 5, 8)
        features = self.down5(x)              # (B, 512, 2, 4)
        
        # --- ATTENTION MECHANISM (optional enhancement) ---
        attention_weights = self.attention(features)  # (B, 1, 2, 4)
        attended_features = features * attention_weights  # Weighted features
        
        # --- GLOBAL POOLING AND CLASSIFICATION ---
        pooled = self.global_avg_pool(attended_features)  # (B, 512, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)       # (B, 512)
        
        output = self.classifier(flattened)               # (B, 1)
        
        return output