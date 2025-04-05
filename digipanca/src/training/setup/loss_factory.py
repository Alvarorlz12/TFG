import torch

from monai.losses import DiceLoss as MONAIDiceLoss

from src.losses.combined_loss import CombinedLoss
from src.losses.focal import FocalLoss
from src.losses.dice import (
    MulticlassDiceLoss,
    WeightedDiceLoss,
    DiceFocalLoss,
    DiceCELoss
)

def get_loss_fn(config):
    """Initialize loss function based on configuration."""
    loss_type = config['training']['loss_function']

    if loss_type == 'MulticlassDiceLoss':
        if config['training'].get('use_monai_loss', False):
            monai_config = config['training']['loss_params']
            # MONAI DiceLoss
            return MONAIDiceLoss(
                to_onehot_y=monai_config.get('to_onehot_y', False),
                softmax=monai_config.get('softmax', False),
                include_background=monai_config.get('include_background', False),
                reduction=monai_config.get('reduction', 'mean'),
            )
        else:
            return MulticlassDiceLoss()
    
    elif loss_type == 'CombinedLoss':
        weights = config['training']['loss_params'].get('weights', None)
        if weights is not None:
            device = torch.device(
                config["training"]["device"] if torch.cuda.is_available() else "cpu"
            )
            weights = torch.tensor(weights).to(device)
        return CombinedLoss(
            alpha=config['training']['loss_params']['alpha'],
            beta=config['training']['loss_params']['beta'],
            class_weights=weights
        )

    elif loss_type == 'FocalLoss':
        return FocalLoss(
            gamma=config['training']['loss_params']['gamma'],
            reduction=config['training']['loss_params']['reduction']
        )

    elif loss_type == 'WeightedDiceLoss':
        return WeightedDiceLoss(
            num_classes=config['training']['loss_params']['num_classes'],
            include_background=config['training']['loss_params']['include_background'],
            reduction=config['training']['loss_params']['reduction']
        )
    
    elif loss_type == 'DiceFocalLoss':
        return DiceFocalLoss(
            include_background=config['training']['loss_params'].get('include_background', False),
            gamma=config['training']['loss_params']['gamma'],
            reduction=config['training']['loss_params']['reduction'],
            lambda_dice=config['training']['loss_params'].get('lambda_dice', 1.0),
            lambda_focal=config['training']['loss_params'].get('lambda_focal', 1.0),
            alpha=config['training']['loss_params'].get('alpha', None)
        )
    
    elif loss_type == 'DiceCELoss':
        return DiceCELoss(
            include_background=config['training']['loss_params'].get('include_background', False),
            reduction=config['training']['loss_params']['reduction'],
            lambda_dice=config['training']['loss_params'].get('lambda_dice', 1.0),
            lambda_ce=config['training']['loss_params'].get('lambda_ce', 1.0)
        )
    
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")