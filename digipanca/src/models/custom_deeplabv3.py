import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.models.resnet import Bottleneck
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

class CustomBottleneck(Bottleneck):
    """Modified ResNet Bottleneck with Dropout after final activation"""
    def __init__(self, inplanes, planes, stride=1, dropout_rate=0.3, **kwargs):
        super().__init__(inplanes, planes, stride=stride, **kwargs)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)  # Dropout after final activation

        return out

class CustomDeepLabV3(nn.Module):
    """Modified DeepLabV3 with customizations:
    - Supports 1-channel input
    - Added dropout in ResNet layer3 and ASPP
    - Custom classifier head
    
    Parameters
    ----------
    num_classes : int
        Number of classes for segmentation.
    dropout_rate : float, optional
        Dropout rate for ResNet layer3 and ASPP.
    pretrained : bool, optional
        Load pretrained weights.
    """
    
    def __init__(self, num_classes, dropout_rate=0.3, pretrained=True):
        super().__init__()

        # Load DeepLabV3 with pretrained weights
        weights = self._get_pretrained_weights(pretrained)
        self.deeplab = deeplabv3_resnet101(weights=weights)

        # Modify input convolution for 1-channel input
        self._adapt_input_channels(pretrained)

        # Add dropout to ResNet layer3 blocks
        self._modify_resnet_layers(dropout_rate)

        # Modify ASPP and classifier
        self._modify_aspp_classifier(dropout_rate, num_classes)

    def _get_pretrained_weights(self, pretrained):
        if pretrained:
            return DeepLabV3_ResNet101_Weights.DEFAULT
        return None

    def _adapt_input_channels(self, pretrained):
        # Replace first convolution layer
        old_conv = self.deeplab.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        if pretrained:
            # Initialize weights from pretrained model
            pretrained_weights = self.deeplab.backbone.conv1.weight.data
            gray_weights = pretrained_weights.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            new_conv.weight.data = gray_weights

        self.deeplab.backbone.conv1 = new_conv

    def _modify_resnet_layers(self, dropout_rate):
        # Replace Bottleneck blocks in layer3 with custom versions
        for i in range(len(self.deeplab.backbone.layer3)):
            block = self.deeplab.backbone.layer3[i]
            if isinstance(block, Bottleneck):
                custom_block = CustomBottleneck(
                    inplanes=block.conv1.in_channels,
                    planes=block.conv1.out_channels,
                    stride=block.stride,
                    dropout_rate=dropout_rate,
                    downsample=block.downsample
                )
                self.deeplab.backbone.layer3[i] = custom_block

    def _modify_aspp_classifier(self, dropout_rate, num_classes):
        # Add dropout to ASPP projection
        aspp_projection = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        )
        self.deeplab.classifier[0].project = aspp_projection

        # Modify intermediate classifier layers
        classifier_layers = [
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        ]
        self.deeplab.classifier[1] = nn.Sequential(*classifier_layers)

        # Replace final classifier
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.deeplab(x)

# if __name__ == "__main__":
#     # TEST: CustomDeepLabV3
#     model = CustomDeepLabV3(num_classes=3, dropout_rate=0.3)
#     input_tensor = torch.randn(2, 1, 256, 256)  # (batch, channels, height, width)
#     output = model(input_tensor)
#     print(f"Output shape: {output['out'].shape}")  # Should be (2, 3, 256, 256)