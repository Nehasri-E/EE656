# Intelligent Condition-Based Monitoring of Air Compressors Using Acoustic Signals 
(EE656 Course Project)

## Repository Structure

### 1. Feature Extraction & Selection
*   `feature_extraction.py` / `feature_extraction_629.py`: Scripts responsible for processing raw signal data and extracting time-domain, frequency-domain, or statistical features.
*   `feature_selectors.py`: Contains methods for dimensionality reduction and selecting the most discriminative features for training.

### 2. Model Implementations
*   `svm_classifier.py`: Implementation of Support Vector Machine (SVM) classification.
*   `rf_implementation.ipynb`: Jupyter notebook detailing the Random Forest (RF) classifier training, hyperparameter tuning, and evaluation.
*   `xbg_implementation.ipynb`: Jupyter notebook implementing the XGBoost (XGB) gradient boosting classifier.

### 3. Experiments & Results
*   `experimentation_1.ipynb`: Initial baseline experiments and model evaluations.
*   `experimentation_2_extendedSet.ipynb`: Extended experiments using an expanded feature set or alternative preprocessing configurations.
*   `Result pickles/`: Directory housing serialized trained models and evaluation outputs for reproducibility.
*   `Report & Presentation/`: Documentation, project reports, and presentation slides detailing the methodology and conclusions.

## Getting Started

### Prerequisites
Ensure you have Python 3.x and the following libraries installed:
```bash
pip install numpy pandas scikit-learn xgboost notebook
