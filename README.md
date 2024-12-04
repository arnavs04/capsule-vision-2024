# Seq2Cure - Capsule Vision 2024 Challenge Submission

## Team Information
- **Team Name**: Seq2Cure
- **Team Members**: Arnav Samal (NIT Rourkela) & Ranya (IGDTUW Delhi)
- **Challenge**: Capsule Vision 2024 - Multi-Class Abnormality Classification for Video Capsule Endoscopy
- **Results**: **Rank**: 5th, **Balanced Accuracy**: 0.8634, **Mean AUC-ROC**: 0.9908

[Link to the Submission Report](https://arxiv.org/abs/2410.18879)
  
## Challenge Overview
The Capsule Vision 2024 Challenge aims to develop AI-based models for multi-class abnormality classification in video capsule endoscopy (VCE) frames. By automating this process, the goal is to reduce the inspection time for gastroenterologists without compromising diagnostic precision. The dataset includes 10 class labels, and teams are evaluated on metrics such as accuracy and AUC.

For more details, visit the [challenge website](https://misahub.in/cv2024.html).

## Directory Structure

```plaintext
capsule-vision-2024/
├── /data
│   ├── training_data_collection.py  # Script for collecting training data
│   └── test_data_collection.txt     # Information on test data collection
├── /logs
│   └── /Model1, /Model2, ...        # Subdirectories named after different models used (store logs)
├── /misc
│   ├── /baseline                    # Contains baseline codebase
│   ├── /info                        # Additional information about the project
│   ├── rough.ipynb                  # A rough Jupyter notebook with preliminary work
│   └── test.py                      # Test script for experimentation
├── /models                          # Empty, to store downloaded models (.pth files)
├── /reports
│   └── /Model1, /Model2, ...        # Subdirectories for reports, corresponding to models in the logs
├── /src
│   ├── /notebooks                   # Jupyter notebooks in PDF format
│   ├── /sample                      # Sample code provided by the organizers
│   ├── data_setup.py                # Script for dataset setup
│   └── ...                          # Other source code files for the project
├── /submission
│   ├── submission_report.pdf        # Submission Report for the Challenge
│   ├── metrics_report.json          # Report detailing the model's evaluation metrics
│   ├── validation_excel.xlsx        # Validation results in Excel format
│   └── Seq2Cure.xlsx                # Final submission in Excel format
├── .gitignore                       # Model Files (*.pth) to be ignored
├── LICENSE                          # License file for the repository
├── README.md                        # Project overview (this file)
└── requirements.txt                 # List of dependencies
```

## Directory Highlights

- **/data**: Contains scripts for collecting and processing training and test datasets.
- **/logs**: Stores logs from model training, organized by the name of the model.
- **/misc**: Includes additional resources such as baseline codebase, papers, and exploratory notebooks.
- **/models**: An empty folder where pre-trained models should be downloaded (to be linked externally).
- **/reports**: Contains reports for each model, tracking performance and experiments.
- **/src**: Includes all source code for the project, including data setup scripts and sample code provided by the challenge organizers.
- **/submission**: The final output files for submission, including performance metrics and the Excel files required by the challenge organizers.

## Data 

The dataset comprises of over 50,000 frames from three public sources and one private dataset, labeled across 10 abnormality classes- Angioectasia, Bleeding, Erosion, Erythema, Foreign Body, Lymphangiectasia, Normal, Polyp, Worms. 

*[Training & validation dataset](https://www.kaggle.com/datasets/arnavs19/capsule-vision-2024-data)*


## Methodology

The project implements a multi-model ensemble approach for video capsule endoscopy frame classification, utilizing three main components:

*For more specific details check the [submission report](submission/submission_report.pdf)*

### Model Architecture
The ensemble incorporates established CNN and transformer architectures:

Traditional CNN Models:
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

Transformer-based Models:
- Vision Transformer (ViT)
- Swin Transformer
- DeiT (Data-efficient Image Transformers)
- BEiT (Bidirectional Encoder Representation from Image Transformers)
- CaiT (Class-Attention in Image Transformers)
- TwinsSVT (Spatially Separable Vision Transformer)
- EfficientFormer

### Training Configuration

Training Parameters:
- Epochs: 20
- Batch Size: 32
- Computing Cores: 4
- Optimizer: AdamW (Learning Rate: 1e-4, Weight Decay: 0.05)
- Model Selection Criterion: (Balanced Accuracy + Mean AUC Score) / 2
- Early Stopping Patience: 5 epochs
- Convergence Tolerance: 1e-4

Class Imbalance Mitigation:
- Weighted Random Sampling based on class frequencies
- Focal Loss implementation

Data Augmentation Pipeline:
- Spatial Transformations:
  - Resize to (224, 224)
  - Random Horizontal Flip (p=0.5)
  - Random Vertical Flip (p=0.3)
  - Random Rotation (±15°)
  - Random Affine (translation: 10%, scale: 0.9-1.1)
  - Random Perspective (distortion scale: 0.2, p=0.5)

- Intensity Transformations:
  - Color Jitter (brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1)
  - Normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  - Random Erasing (p=0.2, scale: 0.02-0.33, ratio: 0.3-3.3)
  - Gaussian Blur (kernel size: 3, p=0.3)

### Inference Methodology

Ensemble Strategy:
- Individual model predictions are generated for each input frame
- Softmax probabilities from each model are averaged
- Final classification is determined by the highest average probability across classes

The implementation uses transfer learning with complete model fine-tuning, adapting pre-trained weights to the specific requirements of capsule endoscopy frame classification.

## Results

The ensemble model's performance was evaluated on the validation dataset, achieving the following key metrics:

- **Balanced Accuracy**: 0.8634
- **Mean AUC-ROC**: 0.9908

Performance across different abnormalities:
1. Highest performing classes:
   - Worms (F1: 0.9927, AUC: 0.9999)
   - Normal (F1: 0.9828, AUC: 0.9960)
   - Ulcer (F1: 0.9570, AUC: 0.9979)

2. Areas for improvement:
   - Erythema (F1: 0.6643, AUC: 0.9892)
   - Polyp (F1: 0.7539, AUC: 0.9858)

Overall model performance:
- Macro-averaged Precision: 0.8666
- Macro-averaged Recall: 0.8634
- Macro-averaged F1-score: 0.8645
- Mean Specificity: 0.9900

For detailed performance metrics including class-wise precision, recall, specificity, and AUC-ROC scores, refer to [metrics_report.json](submission/metrics_report.json).

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1.	Clone the repository:

    ```bash
    git clone https://github.com/your-username/capsule-vision-2024.git
    cd capsule-vision-2024
    ```

2.	Create an environment:
    ```bash
    pip install virtualenv
    virtualenv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.	Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Download Data
   - Run training_data_collection.py
   - Check test_data_collection on how to download testing data
   - For both training and inference phases, we utilized the same dataset, which has been made publicly available on Kaggle through the following links: [training dataset](https://www.kaggle.com/datasets/arnavs19/capsule-vision-2024-data) and [inference dataset](https://www.kaggle.com/datasets/arnavs19/capsule-vision-2020-test).
5. Download Models
   - Download from this [link](https://www.kaggle.com/models/arnavs19/capsule-vision-2024-models)
   - Use *PyTorch, Version 1, Updated* Variation

## License

This repository is licensed under the MIT License, allowing for its use in research and educational purposes.
