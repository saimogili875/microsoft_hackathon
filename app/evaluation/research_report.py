"""
Research Report Generator compiling all 24 required ML research report sections
into markdown and structured JSON.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime, timezone

from app.data.loader import DataLoadReport
from app.data.cleaner import CleaningReport
from app.data.validator import ValidationReport
from app.data.feature_processor import ProcessedDataset
from app.model.evaluation import PerformanceMetrics
from app.explainability.feature_importance import GroupShapSummary
from app.experts.expert_comparison import CaseComparisonResult
from app.evaluation.usefulness import UsefulnessComparisonResult


class ResearchReportGenerator:
    """
    Generates comprehensive 24-section B2B Deal Intelligence Research Report.
    """

    def generate_report(
        self,
        load_report: DataLoadReport,
        clean_report: CleaningReport,
        val_report: ValidationReport,
        dataset: ProcessedDataset,
        metrics: PerformanceMetrics,
        group_shaps: List[GroupShapSummary],
        case_comparisons: List[CaseComparisonResult],
        usefulness_res: UsefulnessComparisonResult,
        output_dir: Path,
    ) -> tuple[str, Path]:

        output_dir.mkdir(parents=True, exist_ok=True)
        report_md_path = output_dir / "research_report.md"
        json_metrics_path = Path("reports/metrics/final_metrics.json")
        json_metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Export JSON metrics
        metrics_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics.to_dict(),
            "usefulness": {
                "prediction_alone": usefulness_res.mean_rating_prediction_alone,
                "prediction_and_explanation": usefulness_res.mean_rating_prediction_and_explanation,
                "difference": usefulness_res.difference_mean_usefulness,
                "evaluations_count": usefulness_res.total_evaluations,
            },
            "feature_groups": {g.feature_group: g.total_shap_importance for g in group_shaps},
        }
        with open(json_metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        # 2. Build 24-Section Markdown Report
        top_cases = case_comparisons[:3]
        sample_cases_text = ""
        for c in top_cases:
            sample_cases_text += f"- **Case {c.case_id} [{c.expert_id}]**: Expert={c.expert_prediction} vs AI={c.ai_prediction} | Agreement={c.prediction_agreement} | Summary: {c.comparison_summary}\n"

        price_feats = ", ".join(dataset.group_to_features.get("PRICE", [])) or "None"
        prod_feats = ", ".join(dataset.group_to_features.get("PRODUCT", [])) or "None"
        org_feats = ", ".join(dataset.group_to_features.get("ORGANIZATION", [])) or "None"
        cust_feats = ", ".join(dataset.group_to_features.get("CUSTOMER", [])) or "None"

        group_shap_text = ""
        for g in group_shaps:
            group_shap_text += f"- **{g.feature_group}**: Total SHAP = {g.total_shap_importance:.4f} ({g.percentage_importance}% importance across {g.feature_count} features)\n"

        report_content = f"""# 🔬 B2B DEAL INTELLIGENCE RESEARCH REPORT
**Generated Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Primary Metric:** Balanced Accuracy  

---

## 1. DATASET SUMMARY
- **Source File:** `{load_report.source_file}`
- **Total Ingested Records:** {load_report.total_records}
- **Raw Win Outcome Count:** {load_report.win_count}
- **Raw Loss Outcome Count:** {load_report.loss_count}

## 2. DATA PROCESSING SUMMARY
- **Initial Input Records:** {clean_report.initial_records}
- **Duplicates Detected & Removed:** {clean_report.duplicates_removed}
- **Invalid Rows Removed:** {clean_report.invalid_rows_removed}
- **Post-Offer Leaked Fields Stripped:** {clean_report.leaked_fields_removed or 'None'}
- **Final Valid Dataset Size:** {clean_report.final_valid_records}

## 3. FEATURE GROUP SUMMARY
- **Total Encoded Features:** {len(dataset.feature_names)}
- **PRICE Group Features Count:** {dataset.feature_group_summary.get('PRICE', 0)}
- **PRODUCT Group Features Count:** {dataset.feature_group_summary.get('PRODUCT', 0)}
- **ORGANIZATION Group Features Count:** {dataset.feature_group_summary.get('ORGANIZATION', 0)}
- **CUSTOMER Group Features Count:** {dataset.feature_group_summary.get('CUSTOMER', 0)}

## 4. PRICE FEATURES
- {price_feats}

## 5. PRODUCT FEATURES
- {prod_feats}

## 6. ORGANIZATION FEATURES
- {org_feats}

## 7. CUSTOMER FEATURES
- {cust_feats}

## 8. TRAINING / TEST METHODOLOGY
- **Train/Test Split:** 80% Train / 20% Test (Stratified)
- **Random State Seed:** 42 (Reproducible)

## 9. XGBOOST CONFIGURATION
- **Algorithm:** XGBoost Classifier (or HistGradientBoosting fallback)
- **Hyperparameters:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42`

## 10. BALANCED ACCURACY (PRIMARY METRIC)
$$\\text{{Balanced Accuracy}} = \\frac{{1}}{{2}} \\left( \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FN}}}} + \\frac{{\\text{{TN}}}}{{\\text{{TN}} + \\text{{FP}}}} \\right) = \\mathbf{{{metrics.balanced_accuracy:.4f}}}$$

## 11. CONFUSION MATRIX
```
                 Predicted WIN    Predicted LOSS
Actual WIN       {metrics.confusion_matrix.get('TP', 0):<15} {metrics.confusion_matrix.get('FN', 0):<15}
Actual LOSS      {metrics.confusion_matrix.get('FP', 0):<15} {metrics.confusion_matrix.get('TN', 0):<15}
```

## 12. PRECISION
- **Precision (WIN):** {metrics.precision:.4f}

## 13. RECALL
- **Overall Recall:** {metrics.recall:.4f}

## 14. F1-SCORE
- **F1-Score:** {metrics.f1_score:.4f}

## 15. WIN / LOSS RECALL DISTRIBUTION
- **WIN Recall (TP / (TP + FN)):** {metrics.win_recall:.4f}
- **LOSS Recall (TN / (TN + FP)):** {metrics.loss_recall:.4f}

## 16. SHAP GLOBAL FEATURE GROUP IMPORTANCE
{group_shap_text}

## 17. SHAP PER-CASE EXPLANATION SAMPLES
{sample_cases_text}

## 18. SALES EXPERT PREDICTIONS
- Preserved independent Sales Expert predictions across evaluated cases.
- **Note:** Expert predictions were strictly excluded from model feature matrix to prevent label leakage.

## 19. Q1–Q5 QUESTIONNAIRE RESULTS SCHEMA
- Q1 (B2B Experience), Q2 (AI Tool Usage), Q3 (Predictive Experience), Q4 (Information Sufficiency), Q5 (Outcome Assessment) implemented and preserved per expert.

## 20. EXPERT VS AI COMPARISON
- Evaluated agreement/disagreement per case.
- Identified shared factors, expert-only factors, and AI-discovered factors without forcing artificial consensus.

## 21. PREDICTION ALONE USEFULNESS
- **Mean Rating:** {usefulness_res.mean_rating_prediction_alone:.2f} / 7.00

## 22. PREDICTION + EXPLANATION USEFULNESS
- **Mean Rating:** {usefulness_res.mean_rating_prediction_and_explanation:.2f} / 7.00

## 23. DIFFERENCE BETWEEN CONDITIONS
- **Usefulness Lift (Prediction + Explanation vs. Prediction Alone):** +{usefulness_res.difference_mean_usefulness:.2f} points
- **Conclusion:** {usefulness_res.summary_text}

## 24. LIMITATIONS
- Synthetic dataset generated for initial pipeline validation; real production deployment requires expanding real CRM/CPQ historical deal records.
- Expert sample size (N={usefulness_res.total_evaluations}) will grow as additional sales reps complete Q1-Q5 evaluation sessions.

---
*Report generated automatically by Deal Intelligence Agent Research Engine.*
"""

        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content, report_md_path
