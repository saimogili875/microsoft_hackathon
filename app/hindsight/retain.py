"""
Hindsight Retain Service converting verified case evaluations into persistent Hindsight memories.
Also provides helper utilities for retaining episodes and market information records.
"""

from typing import List, Dict, Any, Optional
from app.hindsight.client import HindsightClient
from app.hindsight.memory_schema import HindsightDealMemory
from app.explainability.shap_explainer import CaseShapExplanation
from app.experts.expert_comparison import CaseComparisonResult
from app.models.episode import Episode
from app.models.market_info import MarketInfoRecord
from app.models.memory import HindsightMemoryPayload


def format_episode_narrative(episode: Episode) -> str:
    parts = ["[EPISODE]"]
    if episode.situation:
        parts.append(f"Situation:\n{episode.situation}")
    if episode.objection:
        parts.append(f"Objection:\n{episode.objection}")
    if episode.tactic:
        parts.append(f"Tactic:\n{episode.tactic}")
    if episode.customer_reaction:
        parts.append(f"Customer reaction:\n{episode.customer_reaction}")
    if episode.outcome:
        parts.append(f"Outcome:\n{episode.outcome}")
    if episode.why:
        parts.append(f"Why:\n{episode.why}")
    if episode.applies_when:
        parts.append(f"Applies when:\n{episode.applies_when}")
    if episode.did_not_hold_when:
        parts.append(f"Did not hold when:\n{episode.did_not_hold_when}")
    if episode.lesson:
        parts.append(f"Lesson:\n{episode.lesson}")
    if episode.pricing_context:
        parts.append(f"Pricing context:\n{episode.pricing_context}")
    if episode.verified_by_role:
        parts.append(f"Verified by:\n{episode.verified_by_role} ({episode.verification_status})")
    else:
        parts.append(f"Verification status:\n{episode.verification_status}")
    return "\n\n".join(parts)


def retain_episode(episode: Episode, client: HindsightClient) -> HindsightMemoryPayload:
    doc_id = f"episode:{episode.episode_id}"
    content = format_episode_narrative(episode)
    meta = {
        "episode_id": episode.episode_id,
        "deal_id": episode.deal_id,
        "verification_status": episode.verification_status,
        "source_ids": episode.source_ids,
        "causal_confidence": episode.causal_confidence,
    }
    payload = HindsightMemoryPayload(document_id=doc_id, content=content, metadata=meta)
    client.retain(payload)
    return payload


def retain_market_info(market_record: MarketInfoRecord, client: HindsightClient) -> HindsightMemoryPayload:
    m_id = getattr(market_record, "market_id", None) or getattr(market_record, "info_id", None) or getattr(market_record, "record_id", "mkt_info")
    doc_id = f"market_info:{m_id}"
    stmt = getattr(market_record, "statement", None) or getattr(market_record, "description", "")
    content = f"[MARKET INFO]\nTopic: {market_record.topic}\nStatement: {stmt}\nStatus: {market_record.status}"
    meta = market_record.model_dump()
    payload = HindsightMemoryPayload(document_id=doc_id, content=content, metadata=meta)
    client.retain(payload)
    return payload


class HindsightRetainService:
    """
    Retains verified case outcomes and lessons into Hindsight persistent memory.
    """

    def __init__(self, hindsight_client: Optional[HindsightClient] = None):
        self.client = hindsight_client or HindsightClient()

    def retain_case_experience(
        self,
        case_id: str,
        customer_context: str,
        actual_outcome: str,
        shap_explanation: CaseShapExplanation,
        comparison: Optional[CaseComparisonResult] = None,
    ) -> Dict[str, Any]:

        doc_id = f"deal_memory:{case_id}"

        # Extract features per core group from top SHAP positive/negative contributors
        price_feats = [c.feature_name for c in shap_explanation.top_features if c.feature_group == "PRICE"]
        product_feats = [c.feature_name for c in shap_explanation.top_features if c.feature_group == "PRODUCT"]
        org_feats = [c.feature_name for c in shap_explanation.top_features if c.feature_group == "ORGANIZATION"]

        lesson = f"In {actual_outcome} deals with {customer_context}, key influencers were {', '.join(price_feats + product_feats) or 'price & product configuration'}."
        exp_obs = comparison.comparison_summary if comparison else None

        memory = HindsightDealMemory(
            document_id=doc_id,
            case_id=case_id,
            customer_context=customer_context,
            actual_outcome=actual_outcome,
            predicted_outcome=shap_explanation.predicted_label,
            win_probability=shap_explanation.win_probability,
            key_price_factors=price_feats,
            key_product_factors=product_feats,
            key_org_factors=org_feats,
            expert_observation=exp_obs,
            extracted_lesson=lesson,
        )

        payload = memory.to_hindsight_payload()
        res = self.client.retain(payload["document_id"], payload["content"], payload["metadata"])
        return res
