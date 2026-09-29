"""
Episode Extractor module creating structured Episode instances from NormalizedRecord.
Implements strict separation of extraction confidence vs causal confidence.
"""

from typing import List, Optional
from datetime import datetime, timezone
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.extraction.lesson_extractor import LessonExtractor
from app.extraction.memory_classifier import MemoryClassifier


class EpisodeExtractor:
    """
    Extracts structured Episode objects from normalized records.
    Calculates separate extraction_confidence and causal_confidence.
    """

    def __init__(self):
        self.lesson_extractor = LessonExtractor()

    def extract_episode(self, record: NormalizedRecord) -> Episode:
        # Check classification
        category = MemoryClassifier.classify(record)
        if category == "temporary_context":
            # Low confidence extraction for vague context
            ext_conf = 0.3
            causal_conf = 0.1
        else:
            ext_conf, causal_conf = self._evaluate_confidence(record)

        components = self.lesson_extractor.extract_components(record)

        # Source traceability
        source_ids = [ref["raw_id"] for ref in record.source_references if "raw_id" in ref]
        if record.source_id and record.source_id not in source_ids:
            source_ids.append(record.source_id)

        ep = Episode(
            deal_id=record.deal_id,
            situation=components["situation"],
            objection=components["objection"],
            tactic=components["tactic"],
            customer_reaction=components["customer_reaction"],
            outcome=components["outcome"],
            why=components["why"],
            lesson=components["lesson"],
            applies_when=components["applies_when"],
            did_not_hold_when=components["did_not_hold_when"],
            pricing_context=components["pricing_context"],
            timestamp=record.timestamp,
            valid_as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            source_ids=source_ids,
            extraction_confidence=ext_conf,
            causal_confidence=causal_conf,
            verification_status="pending_review",
        )
        return ep

    def _evaluate_confidence(self, record: NormalizedRecord) -> (float, float):
        """
        Calculates extraction_confidence and causal_confidence as distinct metrics.
        
        Extraction confidence:
        - Evaluates field completeness and presence of clear text.
        
        Causal confidence:
        - Evaluates strength of causal attribution.
        - Tactic + WON does NOT imply high causal confidence!
        - If heavy discounting, executive intervention, or multiple confounding factors exist,
          causal confidence is downgraded.
        """
        text = (record.raw_text_content or "").lower()

        # 1. Extraction Confidence Score (0.0 to 1.0)
        ext_score = 0.4  # baseline
        if record.objections:
            ext_score += 0.15
        if record.salesperson_actions:
            ext_score += 0.15
        if record.customer_reactions:
            ext_score += 0.15
        if record.outcome:
            ext_score += 0.15
        ext_confidence = round(min(1.0, ext_score), 2)

        # 2. Causal Confidence Score (0.0 to 1.0)
        # Never automatically give 1.0 just because outcome == won!
        causal_score = 0.5  # baseline hypothesis

        # Confounding factors check
        confounding_factors = [
            "discount", "price cut", "exec sponsor", "ceo intervention",
            "end of quarter", "budget flush", "competitor withdrew", "macro"
        ]
        has_confounding = any(cf in text for cf in confounding_factors)

        if has_confounding:
            # Downgrade causal attribution because win/loss may be due to price cut/timing rather than tactic
            causal_score -= 0.25

        # Check explicit causal markers in text
        if any(marker in text for marker in ["because of tactic", "caused by", "directly resulted in", "key factor"]):
            causal_score += 0.2
        elif record.customer_reactions and record.salesperson_actions:
            causal_score += 0.1

        # Keep within bounds [0.1, 0.95] for unverified extractions
        causal_confidence = round(max(0.1, min(0.90, causal_score)), 2)

        return ext_confidence, causal_confidence
