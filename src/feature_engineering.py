"""Feature engineering functions for fraud detection."""

import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from creation_date."""
    df = df.copy()
    
    # Client tenure features
    df["client_since"] = (2019 - df["creation_date"].dt.year) * 12 - df["creation_date"].dt.month
    df["creation_month"] = df["creation_date"].dt.month
    df["creation_year"] = df["creation_date"].dt.year
    df["is_weekday"] = ((pd.DatetimeIndex(df["creation_date"]).dayofweek) // 5 == 0).astype(float)
    
    return df


def create_consumption_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create consumption-related features."""
    df = df.copy()
    
    # Billing indicators for each consumption level
    for level in range(1, 5):
        df[f"is_billed_level_{level}"] = df[f"consommation_level_{level}"] > 0
    
    return df


def create_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create region-based features."""
    df = df.copy()
    
    df["region"] = df["region"].astype("category")
    df["region_group"] = df["region"].apply(
        lambda x: 100 if x < 100 else 300 if x > 300 else 200
    )
    
    return df


def prepare_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert specified columns to categorical type."""
    categorical_columns = [
        "district", "client_catg", "region", "tarif_type",
        "counter_statue", "counter_code", "reading_remarque",
        "counter_type", "region_group", "is_billed_level_1",
        "is_billed_level_2", "is_billed_level_3", "is_billed_level_4"
    ]
    
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for modeling."""
    columns_to_drop = ["creation_date", "invoice_date"]
    return df.drop(columns=columns_to_drop, errors="ignore")


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = create_time_features(df)
    df = create_consumption_features(df)
    df = create_region_features(df)
    df = prepare_categorical_columns(df)
    df = drop_unnecessary_columns(df)
    
    return df


def get_feature_columns():
    """Return lists of numerical and categorical column names."""
    numerical_columns = [
        "consommation_level_1", "consommation_level_2", 
        "consommation_level_3", "consommation_level_4",
        "old_index", "new_index", "months_number",
        "client_since", "creation_month", "creation_year", "is_weekday"
    ]
    
    categorical_columns = [
        "district", "client_catg", "region", "tarif_type",
        "counter_statue", "counter_code", "reading_remarque",
        "counter_type", "region_group", "is_billed_level_1",
        "is_billed_level_2", "is_billed_level_3", "is_billed_level_4"
    ]
    
    return numerical_columns, categorical_columns