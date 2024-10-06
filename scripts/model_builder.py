import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import timm

model_resnet18 = models.resnet18(pretrained=True)
model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, 10)

model_inception_resnet_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
model_inception_resnet_v2.classifier = nn.Linear(model_inception_resnet_v2.classifier.in_features, 10)

model_mobilenet_v2 = models.mobilenet_v2(pretrained=True)
model_mobilenet_v2.classifier[1] = nn.Linear(model_mobilenet_v2.classifier[1].in_features, 10)

model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
model_swin.head = nn.Linear(model_swin.head.in_features, 10)

model_cswin = timm.create_model('cswin_base_224', pretrained=True)
model_cswin.head = nn.Linear(model_cswin.head.in_features, 10)

model_beit = timm.create_model('beit_base_patch16_224', pretrained=True)
model_beit.head = nn.Linear(model_beit.head.in_features, 10)