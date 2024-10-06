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


# Define Swin Transformer model (using timm)
def model_swin(pretrained=True, num_classes=10):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    # Modify the head to match num_classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


# Define CSwin Transformer model (using timm)
def model_cswin(pretrained=True, num_classes=10):
    model = timm.create_model('cswin_base_224', pretrained=pretrained)
    # Modify the head to match num_classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


# Define BEiT Transformer model (using timm)
def model_beit(pretrained=True, num_classes=10):
    model = timm.create_model('beit_base_patch16_224', pretrained=pretrained)
    # Modify the head to match num_classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model