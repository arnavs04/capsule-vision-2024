import torch
import torch.nn as nn
import timm
import warnings

warnings.filterwarnings('ignore')

# 1. ViT (Vision Transformer)
def model_vit(pretrained=True, num_classes=10):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# 2. Swin Transformer
def model_swin(pretrained=True, num_classes=10):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model

# 3. DeiT (Data-efficient Image Transformers)
def model_deit(pretrained=True, num_classes=10):
    model = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# 4. ConvNeXt
def model_convnext(pretrained=True, num_classes=10):
    model = timm.create_model('convnext_base', pretrained=pretrained)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model

# 5. EfficientNet
def model_efficientnet(pretrained=True, num_classes=10):
    model = timm.create_model('efficientnet_b0', pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# 6. ResNet
def model_resnet(pretrained=True, num_classes=10):
    model = timm.create_model('resnet50', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 7. MobileNetV3
def model_mobilenetv3(pretrained=True, num_classes=10):
    model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# 8. RegNet
def model_regnet(pretrained=True, num_classes=10):
    model = timm.create_model('regnetx_032', pretrained=pretrained)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model

# 9. DenseNet
def model_densenet(pretrained=True, num_classes=10):
    model = timm.create_model('densenet121', pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# 10. Inception v3
def model_inception_v3(pretrained=True, num_classes=10):
    model = timm.create_model('inception_v3', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 11. ResNeXt
def model_resnext(pretrained=True, num_classes=10):
    model = timm.create_model('resnext50_32x4d', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 12. Wide ResNet
def model_wide_resnet(pretrained=True, num_classes=10):
    model = timm.create_model('wide_resnet50_2', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 13. MNASNet
def model_mnasnet(pretrained=True, num_classes=10):
    model = timm.create_model('mnasnet_100', pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# 14. SEResNet50 (Replaces SqueezeNet)
def model_seresnet50(pretrained=True, num_classes=10):
    model = timm.create_model('seresnet50', pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 15. BEiT (Bidirectional Encoder Representation from Image Transformers)
def model_beit(pretrained=True, num_classes=10):
    model = timm.create_model('beit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# 16. CaiT (Class-Attention in Image Transformers)
def model_cait(pretrained=True, num_classes=10):
    model = timm.create_model('cait_s24_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# 17. Twins-SVT (Spatially Separable Vision Transformer)
def model_twins_svt(pretrained=True, num_classes=10):
    model = timm.create_model('twins_svt_base', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# 18. PNASNet
def model_pnasnet(pretrained=True, num_classes=10):
    model = timm.create_model('pnasnet5large', pretrained=pretrained)
    model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
    return model

# 19. XCiT (Cross-Covariance Image Transformers)
def model_xcit(pretrained=True, num_classes=10):
    model = timm.create_model('xcit_medium_24_p8_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# if __name__ == "__main__":
#     # Test the models with random input
#     input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size
    
#     models_to_test = [
#         model_vit, model_swin, model_deit, model_convnext, model_efficientnet,
#         model_resnet, model_mobilenetv3, model_regnet, model_densenet, model_inception_v3,
#         model_resnext, model_wide_resnet, model_mnasnet, 
#         model_seresnet50, 
#         model_beit, model_cait, 
#         model_twins_svt, model_pnasnet, 
#         model_xcit
#     ]
    
#     expected_shape = (1, 10)  # Expected output shape

#     for model_func in models_to_test:
#         model = model_func()
#         output = model(input_tensor)
#         if output.shape != expected_shape:
#             print(f"Model {model_func.__name__} failed with output shape: {output.shape}")
#             break
#         print(f"{model_func.__name__} Output Shape:", output.shape)