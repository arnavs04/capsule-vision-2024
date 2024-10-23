## Models

Specific Models:
- EfficientNet
- ResNet
- MobileNetV3
- RegNet
- DenseNet
- InceptionV4
- ResNeXt
- WideResNet
- MNASNet
- SEResNet50
- ConvNeXt
- ViT (Vision Transformer)
- SwinTransformer
- DeiT (Data-efficient Image Transformers)
- BEiT (Bidirectional Encoder Representation from Image Transformers)
- CaiT (Class-Attention in Image Transformers)
- TwinsSVT (Spatially Separable Vision Transformer)
- EfficientFormer

## Training Techniques
  - Hyperparameters
    - Number of Epochs = 20
    - Batch Size = 32 
    - Number of Cores = 4
  - Training Techniques:
    - Transfer Learning with Full Model Fine-tuning
    - Early Stopping with patience of 5 epochs
    - Saving Best Models according to Combined Score = (Balanced Accuracy + Mean AUC Score) / 2 with tolerance 1e-4
    - AdamW Optimizer with
      - Learning Rate = 1e-4
      - Weight Decay = 0.05
  - Addressing Class Imbalance Issues:
    - **Weighted Random Sampling**: Adjusts the sampling probability of each class based on its frequency to balance class representation during training.
    - **Focal Loss**: Modifies the loss function to focus more on hard-to-classify examples, addressing class imbalance by down-weighting easy examples.
  - Heavy Data Augmentation:
    - **Resize**: Resizes the image to a fixed size of (224, 224) pixels.
    - **RandomHorizontalFlip**: Randomly flips the image horizontally with a probability of 0.5.
    - **RandomVerticalFlip**: Randomly flips the image vertically with a probability of 0.3.
    - **RandomRotation**: Rotates the image randomly within a range of ±15 degrees.
    - **ColorJitter**: Randomly changes the brightness, contrast, saturation, and hue with specified ranges (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1).
    - **RandomAffine**: Applies random affine transformations, such as translation (up to 10% of image size), and scaling (from 90% to 110% of original size).
    - **RandomPerspective**: Randomly applies a perspective transformation with a distortion scale of 0.2 and a probability of 0.5.
    - **Normalize**: Normalizes the image tensor with mean values (0.5, 0.5, 0.5) and standard deviations (0.5, 0.5, 0.5). 
    - **RandomErasing**: Randomly erases a rectangular region of the image with a probability of 0.2, a scale between 0.02 and 0.33 of the image area, and a random aspect ratio between 0.3 and 3.3.
    - **RandomApply (Gaussian Blur)**: Randomly applies Gaussian blur with a kernel size of 3, with a probability of 0.3. 

## Inference & Validation

- Ensemble Learning:
  - Model Ensemble: 
    During inference, an ensemble of multiple models is used to improve prediction accuracy. Each model in the ensemble contributes by making individual predictions, and the final prediction is obtained by averaging the softmax outputs across all models.
    
  - Averaging Strategy: The softmax probabilities from each model are averaged to produce a single ensemble prediction for each sample. This reduces variance and improves robustness in the final classification decision.