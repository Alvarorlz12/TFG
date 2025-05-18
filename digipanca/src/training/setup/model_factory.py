from monai.networks.nets import UNet as MONAIUNet
from monai.networks.layers import Norm

from src.models import (
    UNet,
    CustomDeepLabV3,
    UNet3D
)

def get_model(config):
    """Initialize model based on configuration."""
    model_type = config['model']['type']
    
    if model_type == 'unet':
        if config['model'].get('use_monai', False):
            # Using MONAI UNet
            return MONAIUNet(
                spatial_dims=2,  # 2D images
                in_channels=config['model']['in_channels'],
                out_channels=config['model']['out_channels'],
                channels=config['model'].get('channels', [16, 32, 64, 128, 256]),
                strides=config['model'].get('strides', [2, 2, 2, 2]),
                num_res_units=config['model'].get('num_res_units', 2),
                dropout=config['model'].get('dropout_rate', 0.0),
                norm=Norm.BATCH
            )
        else:
            return UNet(
                in_channels=config['model']['in_channels'],
                out_channels=config['model']['out_channels'],
                init_features=config['model']['init_features']
            )
    
    elif model_type == 'deeplabv3':
        return CustomDeepLabV3(
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate'],
            pretrained=config['model']['pretrained']
        )

    elif model_type == 'unet3d':
        if config['model'].get('use_monai', False):
            return MONAIUNet(
                spatial_dims=3,
                in_channels=config['model']['in_channels'],
                out_channels=config['model']['out_channels'],
                channels=[16, 32, 64, 128, 256],
                strides=[2, 2, 2, 2],
                num_res_units=2,
                norm=Norm.BATCH
            )
        else:
            return UNet3D(
                in_channels=config['model']['in_channels'],
                out_channels=config['model']['out_channels'],
                base_channels=config['model']['base_channels']
            )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")