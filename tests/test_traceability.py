"""
Unit tests for end-to-end case_id traceability across raw data -> processed features -> prediction -> SHAP -> expert -> Hindsight.
"""

import pytest
import pandas as pd
from app.data.loader import DataLoader
from app.data.cleaner import DataCleaner
from app.data.feature_processor import FeatureProcessor
from app.model.train import ModelTrainer
from app.model.predict import ModelPredictor
from app.explainability.shap_explainer import ShapExplainerService
from app.hindsight.retain import HindsightRetainService
from app.hindsight.client import HindsightClient


def test_end_to_end_case_traceability(tmp_path):
    df = pd.DataFrame([
        {"case_id": "CASE-TRACE-001", "price_amount": 50000.0, "product_config_count": 3, "organization_team_size": 4, "customer_company_size_employees": 500, "win_loss_outcome": 1},
        {"case_id": "CASE-TRACE-002", "price_amount": 120000.0, "product_config_count": 10, "organization_team_size": 2, "customer_company_size_employees": 1000, "win_loss_outcome": 0},
        {"case_id": "CASE-TRACE-003", "price_amount": 60000.0, "product_config_count": 4, "organization_team_size": 5, "customer_company_size_employees": 600, "win_loss_outcome": 1},
        {"case_id": "CASE-TRACE-004", "price_amount": 150000.0, "product_config_count": 12, "organization_team_size": 3, "customer_company_size_employees": 1200, "win_loss_outcome": 0},
    ])
    
    cleaner = DataCleaner()
    df_clean, _ = cleaner.clean(df)
    
    processor = FeatureProcessor()
    dataset = processor.process(df_clean)
    
    trainer = ModelTrainer(test_size=0.5, random_state=42)
    train_res = trainer.train(dataset)
    
    predictor = ModelPredictor()
    pred_res = predictor.predict_dataset(train_res.model, train_res.X_test, train_res.test_case_ids)
    
    explainer = ShapExplainerService(train_res.model, dataset.feature_to_group)
    explanations, _ = explainer.explain_dataset(train_res.X_test, train_res.test_case_ids, pred_res.y_pred, pred_res.y_proba)
    
    # Verify case ID preserved across all steps
    target_case_id = str(train_res.test_case_ids.iloc[0])
    assert target_case_id in [pred.case_id for pred in pred_res.case_predictions]
    assert target_case_id in [expl.case_id for expl in explanations]
    
    # Hindsight Retention
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs_trace")
    retain_service = HindsightRetainService(client)
    
    target_expl = [expl for expl in explanations if expl.case_id == target_case_id][0]
    retain_res = retain_service.retain_case_experience(
        case_id=target_case_id,
        customer_context="Acme Logistics Trace Test",
        actual_outcome="WIN",
        shap_explanation=target_expl,
    )
    
    assert retain_res["status"] == "success"
    doc = client.get_document(f"deal_memory:{target_case_id}")
    assert doc is not None
    assert doc.metadata["case_id"] == target_case_id
