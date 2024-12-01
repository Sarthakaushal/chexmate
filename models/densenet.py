import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, freeze_backbone=False):
        super(DenseNetBinaryClassifier, self).__init__()
        
        # Load pretrained DenseNet121 backbone
        if pretrained:
            self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            self.backbone = densenet121(weights=None)
            
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # Get the number of features from the last layer
        num_features = self.backbone.classifier.in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Create new classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        # Pass through new classifier
        output = self.classifier(features)
        return output
    
    def unfreeze_backbone(self):
        """Method to unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def freeze_backbone(self):
        """Method to freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

