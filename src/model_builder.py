import torch.nn as nn
import timm
from torchvision import models


# Define ResNet18 model
def model_resnet18(pretrained=True, num_classes=10):
    model = models.resnet18(pretrained=pretrained)
    # Modify the fully connected layer to match num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Define InceptionResNetV2 model (using timm)
def model_inception_resnet_v2(pretrained=True, num_classes=10):
    model = timm.create_model('inception_resnet_v2', pretrained=pretrained)
    # Modify the classifier to match num_classes
    model.classif = nn.Linear(model.classif.in_features, num_classes)
    return model


# Define MobileNetV2 model
def model_mobilenet_v2(pretrained=True, num_classes=10):
    model = models.mobilenet_v2(pretrained=pretrained)
    # Modify the classifier to match num_classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def model_swin(pretrained=True, num_classes=10):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    original_head_input_features = model.head.fc.in_features
    model.head.fc = nn.Linear(original_head_input_features, num_classes)
    return model


# Define BEiT Transformer model (using timm)
def model_beit(pretrained=True, num_classes=10):
    model = timm.create_model('beit_base_patch16_224', pretrained=pretrained)
    # Modify the head to match num_classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# import torch

# # Test the models with random input
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size

# # Swin Transformer
# swin_model = model_resnet18()
# output = swin_model(input_tensor)
# print("Swin Transformer Output Shape:", output.shape)  # Should be [1, 10]