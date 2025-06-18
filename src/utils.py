"""Utility functions for the fraud detection project."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import os


def load_data(data_path: str = "data/raw/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all datasets from the specified path."""
    try:
        df_client = pd.read_csv(os.path.join(data_path, "client_train.csv"))
        df_invoice = pd.read_csv(os.path.join(data_path, "invoice_train.csv"))
        df_client_test = pd.read_csv(os.path.join(data_path, "client_test.csv"))
        df_invoice_test = pd.path.join(data_path, "invoice_test.csv"))
        
        return df_client, df_invoice, df_client_test, df_invoice_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure data files are in the data/raw/ directory")
        raise


def save_processed_data(df: pd.DataFrame, filename: str, data_path: str = "data/processed/"):
    """Save processed dataframe to specified path."""
    os.makedirs(data_path, exist_ok=True)
    df.to_csv(os.path.join(data_path, filename), index=False)
    print(f"Saved {filename} to {data_path}")


def basic_info_display(df: pd.DataFrame, name: str = "DataFrame"):
    """Display basic information about a dataframe."""
    print(f"\n=== {name} Info ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")