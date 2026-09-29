"""
Unit tests for data loading, cleaning, leakage protection, and feature processing.
"""

import pytest
import pandas as pd
from app.data.loader import DataLoader
from app.data.cleaner import DataCleaner
from app.data.validator import DataValidator
from app.data.feature_processor import FeatureProcessor


def test_data_cleaner_leakage_protection():
    df = pd.DataFrame([
        {"case_id": "C1", "price_amount": 100, "post_deal_churn": 1, "win_loss_outcome": 1},
        {"case_id": "C2", "price_amount": 200, "post_deal_churn": 0, "win_loss_outcome": 0},
    ])
    cleaner = DataCleaner()
    df_clean, report = cleaner.clean(df)
    
    assert "post_deal_churn" not in df_clean.columns
    assert "post_deal_churn" in report.leaked_fields_removed


def test_feature_processor_group_mapping():
    df = pd.DataFrame([
        {"case_id": "C1", "price_amount": 100, "product_config": "X", "organization_region": "US", "win_loss_outcome": 1},
    ])
    processor = FeatureProcessor()
    dataset = processor.process(df)
    
    assert "price_amount" in dataset.group_to_features["PRICE"]
    assert dataset.feature_to_group["price_amount"] == "PRICE"
    assert "organization_region_US" in dataset.feature_to_group
    assert dataset.feature_to_group["organization_region_US"] == "ORGANIZATION"
