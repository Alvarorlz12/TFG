import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class Hola(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        """
        Initialize the DeepLabV3 model with a ResNet-50 backbone.

        Parameters
        ----------
        num_classes : int
            Number of classes in the dataset.
        pretrained : bool, optional
            Use ImageNet pre-trained weights.
        freeze_backbone : bool, optional
            Freeze the backbone weights.
        """
        super(Hola, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained)
        
        # Replace the classifier with a new one
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier = DeepLabHead(in_channels, num_classes)
        
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.model(x)["out"]