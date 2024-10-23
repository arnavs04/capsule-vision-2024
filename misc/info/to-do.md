# To Do

- Better Performance
  - New 19 Models
  - Training on Basis of Balanced Accuracy and MeanAUC Score
  - 
- Rechecking Submission Format
- Organize Repo with readme especially
- Submission Report
- Training [DONE]
- Inference (Ensembling) [DONE]
- Early Stopping, Saving Model with Better Techniques [DONE]
- Handling Class Imbalance [DONE]
- Data Augmentation, Resize [DONE]

# Imp. Details
- Evaluation Metrics (<u>Mean Auc</u>, F1 Score, ROC-AUC)
- 10 Class Labels (Multi-Class Classification)

## Models
*can be pre-trained, ensemble, and deep learning methods*
- CNNs (ResNet18, InceptionNetResNetv2, MobileNetv2)
- Vision Transformers (Swin, BEiT/DeiT)
- Research Specific (Open up research papers to find transformers made exactly for specific tasks like this)

## Training Techniques
- Knowledge Distillation (If another trained model found)
- Iterative K-fold (4-5 Folds)
- Self-Supervised Pre-training (Masked Auto-Encoding, Contrastive Learning)
- Optimization Strategies (Cosine Learning Rate Schedule, Warmup, Layer-wise Learning Rate Decay, Early-stopping)

# Others

## Dealing with Class Imbalances

### 1. Data-level Approaches

- **Oversampling**: Increase the number of samples in minority classes.
  - Random Oversampling
  - SMOTE (Synthetic Minority Over-sampling Technique)
- **Undersampling**: Reduce the number of samples in majority classes.
  - Random Undersampling
  - Tomek Links
- **Hybrid Methods**: Combine oversampling and undersampling.
  - SMOTETomek
  - SMOTEENN

### 2. Algorithm-level Approaches

- **Class Weighting**: Assign higher weights to minority classes in the loss function.
  - Inverse Class Frequency
  - Effective Number of Samples
- **Focal Loss**: Dynamically adjust the loss to focus on hard examples and down-weight easy ones.
- **LDAM Loss**: Large Margin Loss with adjusted margins based on class frequencies.

### 3. Ensemble Methods

- **Balanced Bagging**: Create multiple balanced subsets and train a model on each.