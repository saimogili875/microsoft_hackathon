"""
Deal Intelligence Agent Research Prototype Master Orchestrator.
Executes complete 12-phase pipeline:
Raw CRM/CPQ Data -> Data Cleaning -> Feature Engineering (Price/Product/Org/Customer)
-> XGBoost Model -> Balanced Accuracy & Confusion Matrix -> SHAP Explainability
-> Sales Expert Evaluation (Q1-Q5) -> Expert vs AI Comparison
-> Prediction Alone vs Prediction + Explanation Usefulness Evaluation
-> Hindsight Retention -> 24-Section Research Report Generation.
"""

import sys
from pathlib import Path
import json

from app.data.loader import DataLoader
from app.data.cleaner import DataCleaner
from app.data.validator import DataValidator
from app.data.feature_processor import FeatureProcessor
from app.model.train import ModelTrainer
from app.model.predict import ModelPredictor
from app.model.evaluation import ModelEvaluator
from app.explainability.shap_explainer import ShapExplainerService
from app.explainability.feature_importance import FeatureImportanceAnalyzer
from app.experts.expert_predictions import ExpertPredictionManager
from app.experts.expert_comparison import ExpertAiComparisonEngine
from app.evaluation.usefulness import UsefulnessEvaluator
from app.evaluation.research_report import ResearchReportGenerator
from app.hindsight.retain import HindsightRetainService


def main():
    print("==================================================================")
    print("🚀 B2B DEAL INTELLIGENCE AGENT — RESEARCH PROTOTYPE PIPELINE")
    print("==================================================================")

    # PATHS
    data_file = Path("data/raw/deals_dataset.csv")
    expert_file = Path("data/experts/expert_evaluations.json")
    reports_dir = Path("reports/final_report")
    shap_dir = Path("reports/shap")

    # PHASE 1: DATA LOADING
    print("\n[PHASE 1] Loading CRM/CPQ Dataset...")
    loader = DataLoader()
    df_raw, load_report = loader.load_file(data_file)
    print(f"-> Ingested {load_report.total_records} raw deal records ({load_report.win_count} WIN / {load_report.loss_count} LOSS).")

    # PHASE 2: DATA CLEANING & LEAKAGE PROTECTION
    print("\n[PHASE 2] Cleaning Data & Enforcing Post-Offer Leakage Protection...")
    cleaner = DataCleaner()
    df_clean, clean_report = cleaner.clean(df_raw)
    print(f"-> Final valid records: {clean_report.final_valid_records} (Duplicates removed: {clean_report.duplicates_removed}).")

    # PHASE 3: FEATURE ENGINEERING & CORE FEATURE GROUPS
    print("\n[PHASE 3] Feature Engineering & Core Feature Group Mapping...")
    validator = DataValidator()
    val_report = validator.validate(df_clean)
    processor = FeatureProcessor()
    dataset = processor.process(df_clean)
    print(f"-> Processed {len(dataset.feature_names)} encoded features across Core Groups:")
    for grp, count in dataset.feature_group_summary.items():
        print(f"   * {grp}: {count} features")

    # PHASE 4: XGBOOST MODEL TRAINING
    print("\n[PHASE 4] Training XGBoost Classifier (Reproducible Random State = 42)...")
    trainer = ModelTrainer(test_size=0.20, random_state=42)
    train_res = trainer.train(dataset)
    print(f"-> Fitted {train_res.model.model_type} model on {len(train_res.X_train)} train cases.")

    # PHASE 5: MODEL INFERENCE & EVALUATION (BALANCED ACCURACY)
    print("\n[PHASE 5] Evaluating Model Performance & Computing Balanced Accuracy...")
    predictor = ModelPredictor()
    pred_res = predictor.predict_dataset(train_res.model, train_res.X_test, train_res.test_case_ids)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(train_res.y_test.values, pred_res.y_pred)
    print(f"-> 🎯 BALANCED ACCURACY: {metrics.balanced_accuracy:.4f}")
    print(f"-> Precision: {metrics.precision:.4f} | Recall: {metrics.recall:.4f} | F1: {metrics.f1_score:.4f}")
    print(f"-> WIN Recall: {metrics.win_recall:.4f} | LOSS Recall: {metrics.loss_recall:.4f}")
    print(f"-> Confusion Matrix: {metrics.confusion_matrix}")

    # PHASE 6: SALES EXPERT PREDICTIONS & Q1-Q5 SCHEMA
    print("\n[PHASE 6] Loading Sales Expert Evaluations (Q1-Q5)...")
    expert_mgr = ExpertPredictionManager()
    expert_recs = expert_mgr.load_file(expert_file)
    print(f"-> Loaded {len(expert_recs)} independent Sales Expert evaluation responses.")

    # PHASE 7: SHAP EXPLAINABILITY
    print("\n[PHASE 7] Generating SHAP Explanations & Global Feature Importance...")
    shap_service = ShapExplainerService(train_res.model, dataset.feature_to_group)
    case_explanations, shap_values = shap_service.explain_dataset(
        train_res.X_test, train_res.test_case_ids, pred_res.y_pred, pred_res.y_proba
    )

    feat_analyzer = FeatureImportanceAnalyzer()
    feat_importances, group_shaps = feat_analyzer.analyze(train_res.X_test, shap_values, dataset.feature_to_group)

    print("-> Top Core Feature Groups by SHAP Importance:")
    for g in group_shaps:
        print(f"   * {g.feature_group}: SHAP Total = {g.total_shap_importance:.4f} ({g.percentage_importance}%)")

    shap_service.generate_shap_plots(train_res.X_test, shap_values, shap_dir)

    # PHASE 8: EXPERT VS AI COMPARISON
    print("\n[PHASE 8] Executing Expert vs. AI + SHAP Comparison Layer...")
    comparison_engine = ExpertAiComparisonEngine()
    case_comparisons = []
    case_expl_dict = {ce.case_id: ce for ce in case_explanations}
    case_pred_dict = {cp.case_id: cp for cp in pred_res.case_predictions}

    for exp_rec in expert_recs:
        cid = exp_rec.case_id
        if cid in case_expl_dict and cid in case_pred_dict:
            comp_res = comparison_engine.compare_case(case_pred_dict[cid], case_expl_dict[cid], exp_rec)
            case_comparisons.append(comp_res)

    print(f"-> Evaluated {len(case_comparisons)} Expert vs AI case comparisons.")
    if case_comparisons:
        agree_count = sum(1 for c in case_comparisons if c.prediction_agreement)
        print(f"   * Prediction Agreement Rate: {agree_count}/{len(case_comparisons)} ({agree_count/len(case_comparisons):.1%})")

    # PHASE 9: PREDICTION ALONE VS PREDICTION + EXPLANATION EVALUATION
    print("\n[PHASE 9] Evaluating Prediction Alone vs. Prediction + Explanation Usefulness...")
    usefulness_eval = UsefulnessEvaluator()
    usefulness_res = usefulness_eval.evaluate_records(expert_recs)
    print(f"-> {usefulness_res.summary_text}")

    # PHASE 10: HINDSIGHT MEMORY RETENTION
    print("\n[PHASE 10] Retaining Verified Case Experiences in Hindsight Memory...")
    retain_service = HindsightRetainService()
    retained_count = 0
    for comp in case_comparisons[:10]:  # Retain top case experiences
        cid = comp.case_id
        if cid in case_expl_dict:
            actual = "WIN" if train_res.y_test.iloc[0] == 1 else "LOSS"
            retain_service.retain_case_experience(
                case_id=cid,
                customer_context=f"Deal Case {cid}",
                actual_outcome=actual,
                shap_explanation=case_expl_dict[cid],
                comparison=comp,
            )
            retained_count += 1
    print(f"-> Retained {retained_count} verified deal case experiences in Hindsight memory.")

    # PHASE 11 & 12: RESEARCH REPORT GENERATION
    print("\n[PHASE 12] Generating Final 24-Section Machine Learning Research Report...")
    report_gen = ResearchReportGenerator()
    report_md, report_path = report_gen.generate_report(
        load_report=load_report,
        clean_report=clean_report,
        val_report=val_report,
        dataset=dataset,
        metrics=metrics,
        group_shaps=group_shaps,
        case_comparisons=case_comparisons,
        usefulness_res=usefulness_res,
        output_dir=reports_dir,
    )
    print(f"-> Saved master research report to {report_path}")

    print("\n==================================================================")
    print("✅ PROTOTYPE PIPELINE SUCCESSFULLY EXECUTED!")
    print("==================================================================")

if __name__ == "__main__":
    main()
