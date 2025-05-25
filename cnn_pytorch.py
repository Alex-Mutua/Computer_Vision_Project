import torch
import torchvision.models as models

def get_pretrained_model(num_classes=4):
    """Initialize a pretrained ResNet34 model for brain tumor classification"""
    # Load pretrained ResNet34
    model = models.resnet34(weights='IMAGENET1K_V1')
    # Freeze base model layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final fully connected layer
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model