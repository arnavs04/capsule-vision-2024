import torch
import torch.nn as nn
import timm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def model_swin(pretrained=True, num_classes=10):
    # Load the Swin Transformer model with pretrained weights
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)

    # Get the original input features of the head
    original_head_input_features = model.head.in_features

    # # Replace the original classification head with a new linear layer
    # model.head = nn.Linear(original_head_input_features, num_classes)

    # # Add global average pooling layer before the classification head
    # model.avgpool = nn.AdaptiveAvgPool2d(1)  # This will pool to 1x1

    return model

# Test the model
input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size
swin_model = model_swin()

# Forward pass to get the output
output = swin_model(input_tensor)

# Print the shape of the output
# print("Swin Transformer Output Shape:", output.shape)  # Expected: [1, num_classes]
print(swin_model)