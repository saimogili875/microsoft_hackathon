"""
End-to-End Demo Scenario Script: ABC Logistics Deal Intelligence Loop.

Demonstrates the 10-step Salesperson User Journey:
1. Selects customer 'ABC Logistics'.
2. Loads Customer Relationship Intelligence (CRM Factual + Hindsight Persistent Memory).
3. Executes Historical Learning & Retention (ARCH-1 Hindsight RETAIN).
4. Live Interaction Stream: Receives new objection & pricing signal.
5. Real-Time Change Detection: Competitor price shift (₹10L -> ₹7L) marked PENDING VERIFICATION.
6. Information Retrieval (ARCH-2 Hindsight RECALL): Retrieves past win tactic (phased deployment).
7. AI Reasoning & Sales Intelligence: Synthesizes past experience + live signals.
8. ML Win/Loss Prediction & SHAP Explanation: Predicts WIN probability & feature group contributions.
9. Human Verification: Account Executive confirms new ₹7L competitor pricing fact.
10. Final Hindsight RETAIN: Retains new verified memory into persistent layer.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.pipeline import DealIntelligencePipeline
from app.models.live_interaction import LiveTranscriptChunk
from app.models.change_event import ClientState
from app.models.raw_data import RawRecord
from app.model.predict import ModelPredictor
from app.explainability.shap_explainer import ShapExplainerService
from app.experts.expert_comparison import ExpertAiComparisonEngine
from app.experts.expert_predictions import ExpertEvaluationRecord


def run_demo():
    print("=" * 80)
    print("🚀 DEAL INTELLIGENCE AGENT — END-TO-END DEMO SCENARIO")
    print("Customer: ABC Logistics")
    print("=" * 80)

    pipeline = DealIntelligencePipeline()

    # -------------------------------------------------------------------------
    # STEP 1: LOAD HISTORICAL DEAL & RETAIN IN HINDSIGHT (ARCH-1)
    # -------------------------------------------------------------------------
    print("\n--- STEP 1 & 2: HISTORICAL RETENTION & CUSTOMER SELECTION ---")
    
    # Simulate ingesting historical winning deal experience for ABC Logistics
    hist_raw = RawRecord(
        source_id="demo_abc_hist_001",
        source_type="crm_record",
        raw_content="""
        Client: ABC Logistics (Deal D-ABC-100)
        Outcome: Closed Won
        Initial Price: ₹12L quote, agreed at ₹10L
        Objection: Implementation cost too high and integration complexity.
        Tactic Used: Offered phased rollout pilot with pre-built connectors.
        Reaction: Customer agreed to pilot and signed phase 1.
        Lesson: Phased rollout overcomes implementation cost friction for logistics clients.
        Competitor: Competitor X (offered ₹10L).
        """,
        metadata={"client_id": "ABC Logistics", "deal_id": "D-ABC-100"}
    )
    
    # Process historical record through pipeline
    norm_rec = pipeline.normalizer.normalize(hist_raw)
    ext_res = pipeline.groq_extractor.extract_from_record(norm_rec)
    
    if ext_res.episodes:
        hist_ep = pipeline.groq_extractor.convert_to_episode(ext_res.episodes[0], norm_rec)
        pipeline.verification_service.save_pending(hist_ep)
        # Verify & retain into Hindsight
        pipeline.verify_and_retain(hist_ep.episode_id, action="confirm", role="account_executive")
        print(f"✅ Retained Historical Episode into Hindsight: {hist_ep.episode_id}")
    else:
        print("⚠️ Used fallback memory retention.")
        pipeline.hindsight_client.retain(
            content="ABC Logistics deal D-ABC-100 won: Phased rollout pilot overcame implementation cost objection against Competitor X (₹10L).",
            metadata={"client_id": "ABC Logistics", "deal_id": "D-ABC-100", "verification_status": "confirmed"}
        )

    # -------------------------------------------------------------------------
    # STEP 3: GENERATE CUSTOMER RELATIONSHIP INTELLIGENCE
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: CUSTOMER RELATIONSHIP INTELLIGENCE REPORT ---")
    client_state = ClientState(
        deal_id="D-ABC-200",
        customer_context="ABC Logistics — Enterprise Freight & Supply Chain",
        stakeholders=["John Smith (VP Ops)", "Sarah Jenkins (CTO)"],
        tools_used=["Legacy TMS", "Excel"],
        competitors=["Competitor X"],
        active_objections=["Implementation cost concern"],
        resolved_objections=["Integration complexity"],
        commitments=["Share modular deployment timeline"],
    )
    
    report = pipeline.report_generator.generate_client_relationship_report(
        client_state, historical_records=[norm_rec]
    )
    print(report)

    # -------------------------------------------------------------------------
    # STEP 4: LIVE INTERACTION & SIGNAL EXTRACTION
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: LIVE INTERACTION STREAM ---")
    chunk1 = LiveTranscriptChunk(
        session_id="live_abc_99",
        speaker="Customer (John Smith)",
        text="We are currently evaluating your platform, but our main concern is implementation cost.",
        timestamp="10:00:15"
    )
    chunk2 = LiveTranscriptChunk(
        session_id="live_abc_99",
        speaker="Customer (John Smith)",
        text="Competitor X is now offering us a lower price of ₹7L.",
        timestamp="10:01:30"
    )

    res1 = pipeline.process_live_chunk(chunk1)
    res2 = pipeline.process_live_chunk(chunk2)

    print(f"Detected Signals Chunk 1: {res1['extracted_signals']}")
    print(f"Detected Signals Chunk 2: {res2['extracted_signals']}")

    # -------------------------------------------------------------------------
    # STEP 5: CHANGE DETECTION (OLD ₹10L vs NEW ₹7L -> PENDING VERIFICATION)
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: CHANGE DETECTION ---")
    changes = pipeline.detect_changes(client_state, norm_rec)
    print(f"Detected Changes Count: {len(changes)}")
    for chg in changes:
        print(f" - [{chg.category.value}] Old: {chg.old_information} | New: {chg.new_evidence} | Status: {chg.status}")

    # -------------------------------------------------------------------------
    # STEP 6 & 7: ARCH-2 RETRIEVAL & GROQ REASONING
    # -------------------------------------------------------------------------
    print("\n--- STEP 6 & 7: ARCH-2 INFORMATION RETRIEVAL & AI REASONING ---")
    arch2_res = pipeline.arch2_query(
        query="What objections did ABC Logistics raise previously and what tactics worked?",
        client_id="ABC Logistics",
        deal_id="D-ABC-100"
    )

    print(f"Memories Found: {arch2_res.retrieved_memory_count}")
    print(f"AI Evidence-Grounded Answer:\n{arch2_res.answer}")

    # -------------------------------------------------------------------------
    # STEP 8: ML SALES WIN/LOSS PREDICTION & SHAP EXPLAINABILITY
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: ML WIN/LOSS PREDICTION & SHAP ---")
    from app.data.loader import DataLoader
    from app.data.cleaner import DataCleaner
    from app.data.validator import DataValidator
    from app.data.feature_processor import FeatureProcessor
    from app.model.train import ModelTrainer

    loader = DataLoader()
    raw_df, _ = loader.load_file("data/raw/deals_dataset.csv")
    clean_df, _ = DataCleaner().clean(raw_df)
    val_report = DataValidator().validate(clean_df)
    dataset = FeatureProcessor().process(clean_df)

    trainer = ModelTrainer(random_state=42)
    train_res = trainer.train(dataset)

    predictor = ModelPredictor()
    pred_res = predictor.predict_dataset(train_res.model, train_res.X_test, train_res.test_case_ids)

    sample_pred = pred_res.case_predictions[0]
    print(f"Sample Case ID: {sample_pred.case_id}")
    print(f"Prediction Outcome: {sample_pred.predicted_label}")
    print(f"WIN Probability: {sample_pred.win_probability * 100:.1f}%")
    print(f"LOSS Probability: {sample_pred.loss_probability * 100:.1f}%")
    print(f"Prediction Confidence: {sample_pred.confidence * 100:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 9 & 10: HUMAN VERIFICATION & FINAL HINDSIGHT RETENTION
    # -------------------------------------------------------------------------
    print("\n--- STEP 9 & 10: HUMAN VERIFICATION & PERSISTENT MEMORY RETAIN ---")
    pending = pipeline.verification_service.get_pending_episodes()
    print(f"Pending Verification Queue Length: {len(pending)}")
    
    if pending:
        target_ep = pending[0]
        v_res = pipeline.verify_and_retain(
            episode_id=target_ep.episode_id,
            action="confirm",
            role="account_executive"
        )
        print(f"✅ Verified & Retained Episode '{target_ep.episode_id}' -> Status: {v_res['verification_status']}")
    
    print("\n" + "=" * 80)
    print("🎉 END-TO-END DEMO SCENARIO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
