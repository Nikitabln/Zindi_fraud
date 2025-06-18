# Fraud Detection in Electricity and Gas Consumption

A machine learning solution for detecting fraud in electricity and gas consumption data from the [Zindi Competition](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge).

## 🎯 Project Overview

This project implements a comprehensive fraud detection system using LightGBM with advanced preprocessing techniques including SMOTE for handling class imbalance.


## 📁 Project Structure
```
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   └── 01_complete_analysis.ipynb  # Your original notebook
├── src/                    # Source code modules
│   ├── utils.py           # Utility functions
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── modeling.py
├── data/                   # Data files (not tracked in git)
│   ├── raw/               # Original data files
│   └── processed/         # Cleaned data
├── models/                 # Saved models
├── results/               # Results and figures
│   ├── figures/
│   └── submissions/
├── config/                # Configuration files
│   └── config.yaml
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
git clone <your-repo-url>
cd Zindi_fraud
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data Setup
1. Download data from the [Zindi competition page](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge/data)
2. Extract CSV files to `data/raw/` directory

### Usage
```python
# In a notebook or script
import sys
sys.path.append('src')

from utils import load_data
from data_preprocessing import merge_datasets, preprocess_data
from feature_engineering import engineer_all_features

# Load and process data
df_client, df_invoice, df_client_test, df_invoice_test = load_data()
df, df_test = merge_datasets(df_client, df_invoice, df_client_test, df_invoice_test)
df = preprocess_data(df)
df = engineer_all_features(df)
```

## 📊 Key Features

- **Modular Code Structure**: Clean separation of data processing, feature engineering, and modeling
- **Advanced Feature Engineering**: Time-based, consumption-based, and regional features  
- **Class Imbalance Handling**: SMOTE oversampling technique
- **Robust Preprocessing**: Automated data cleaning and standardization
- **Hyperparameter Optimization**: RandomizedSearchCV for model tuning

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning and preprocessing
- **LightGBM**: Gradient boosting framework
- **Imbalanced-learn**: SMOTE for handling imbalanced data
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive development

## 📈 Results

- **ROC-AUC**: 0.77+
- **Model**: LightGBM with SMOTE oversampling
- **Features**: 25+ engineered features


## 📄 License

This project is for educational purposes as part of the Zindi competition.