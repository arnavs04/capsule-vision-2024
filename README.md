# Seq2Cure - Capsule Vision 2024 Challenge Submission

## Team Information
- **Team Name**: Seq2Cure
- **Team Members**: Arnav Samal (NIT Rourkela) & Ranya (IGDTUW)
- **Challenge**: Capsule Vision 2024 - Multi-Class Abnormality Classification for Video Capsule Endoscopy

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
│   ├── /papers                      # Relevant research papers
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
│   ├── metrics_report.json          # Report detailing the model's evaluation metrics
│   ├── validation_excel.xlsx        # Validation results in Excel format
│   └── Seq2Cure.xlsx                # Final submission in Excel format
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

## Methodology

## Results

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