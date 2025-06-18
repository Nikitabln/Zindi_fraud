"""Data preprocessing functions for fraud detection."""

import pandas as pd
import numpy as np
from typing import Tuple


def merge_datasets(df_client: pd.DataFrame, df_invoice: pd.DataFrame, 
                  df_client_test: pd.DataFrame, df_invoice_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge client and invoice datasets."""
    # Convert date columns
    for df in [df_invoice, df_invoice_test]:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    
    for df in [df_client, df_client_test]:
        df["creation_date"] = pd.to_datetime(df["creation_date"], format="mixed")
    
    # Merge datasets
    df = pd.merge(df_client, df_invoice, on="client_id", how="left")
    df_test = pd.merge(df_client_test, df_invoice_test, on="client_id", how="left")
    
    return df, df_test


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Fix column name typos."""
    if 'disrict' in df.columns:
        df = df.rename(columns={"disrict": "district"})
    return df


def clean_client_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Clean client_id column by extracting numeric part."""
    if df['client_id'].dtype == 'object':
        df["client_id"] = df["client_id"].apply(lambda x: x.split("_")[-1]).astype(int)
    return df


def clean_counter_statue(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize counter_statue values."""
    counter_statue_map = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        769: 5, "0": 0, "5": 5, "1": 1, "4": 4,
        "A": 0, 618: 5, 269375: 5, 46: 5, 420: 5,
    }
    df["counter_statue"] = df["counter_statue"].map(counter_statue_map)
    return df


def clean_counter_type(df: pd.DataFrame) -> pd.DataFrame:
    """Convert counter_type to binary encoding."""
    df["counter_type"] = df["counter_type"].map({"GAZ": 0, "ELEC": 1})
    return df


def clean_months_number(df: pd.DataFrame, col: str = "months_number") -> pd.DataFrame:
    """Clean months_number column by capping at 12 and interpolating."""
    df = df.copy()
    df.loc[df[col] > 12, col] = None
    df[col] = df[col].interpolate(method="linear")
    df[col] = df[col].round().clip(1, 12).astype("Int64")
    return df


def remove_reading_remarque_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove specific reading_remarque outliers."""
    return df[~df["reading_remarque"].isin([203, 207, 413])]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps."""
    df = clean_column_names(df)
    df = clean_client_ids(df)
    df = clean_counter_statue(df)
    df = clean_counter_type(df)
    df = clean_months_number(df)
    
    # Only remove outliers from training data (check if target exists)
    if 'target' in df.columns:
        df = remove_reading_remarque_outliers(df)
    
    return df