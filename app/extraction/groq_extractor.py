"""
Groq LLM Semantic Extractor module for ARCH-1.
Analyzes raw/normalized sales records and transforms them into grounded GroqExtractionResult JSON.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
import httpx
from config.settings import settings
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.models.groq_extraction_schema import GroqExtractionResult, ExtractedEpisodeSchema

logger = logging.getLogger(__name__)

GROQ_EXTRACTION_SYSTEM_PROMPT = """You are the RAW SALES DATA → MEMORY EXTRACTION ENGINE for a B2B Deal Intelligence Agent.

Your ONLY responsibility is to analyze raw sales information and transform it into structured, evidence-grounded knowledge that can later be reviewed by a human and stored in Hindsight.

You are NOT the final sales advisor.
You are NOT generating a sales report.
You are NOT recommending what the salesperson should do.
You are NOT allowed to invent missing information.

Your output will be consumed programmatically by Python.

CORE RULE:
Extract only information supported by the source.

NEVER:
- invent facts
- guess missing information
- infer an unstated customer reaction
- infer an unstated outcome
- infer causality without evidence
- turn a salesperson opinion into an established fact
- create competitors that were not mentioned
- create prices that were not mentioned
- create stakeholders that were not mentioned
- create lessons unsupported by evidence

Missing scalar information = null.
Missing array information = [].

SEPARATE:
FACT: Directly supported by source.
CLAIM: Someone reported/believed something.
INFERENCE: Something that could be concluded but is not directly supported.
Do not convert CLAIM or INFERENCE into FACT.

CAUSALITY RULE:
Never assume: TACTIC + WON = TACTIC CAUSED WIN or TACTIC + LOST = TACTIC CAUSED LOSS.
Maintain extraction_confidence and causal_confidence as separate values.

TRACEABILITY:
Every extracted episode must preserve source_id, deal_id, episode_id, timestamp, evidence_spans.

OUTPUT ONLY VALID JSON MATCHING THIS EXACT SCHEMA:
{
  "extraction_metadata": {
    "source_id": "",
    "source_type": "",
    "source_timestamp": null,
    "extraction_version": "1.0",
    "overall_extraction_confidence": 0.0
  },
  "episodes": [
    {
      "episode_id": null,
      "deal_id": null,
      "client_id": null,
      "context": {
        "stage": null,
        "situation": null,
        "goals": [],
        "pain_points": [],
        "requirements": []
      },
      "customer": {
        "objections": [],
        "concerns": [],
        "reaction": null,
        "preferences": [],
        "priorities": []
      },
      "salesperson": {
        "tactic": null,
        "actions_taken": [],
        "commitments": []
      },
      "outcome": {
        "status": null,
        "explicit_reason": null,
        "next_step": null
      },
      "competition": [],
      "commercial": {
        "pricing_context": null,
        "quoted_price": null,
        "budget": null,
        "discount": null
      },
      "stakeholders": [],
      "products_and_tools": [],
      "learning": {
        "lesson": null,
        "applies_when": [],
        "did_not_hold_when": []
      },
      "temporal": {
        "event_timestamp": null,
        "valid_as_of": null
      },
      "provenance": {
        "source_ids": [],
        "evidence_spans": []
      },
      "confidence": {
        "extraction_confidence": 0.0,
        "causal_confidence": 0.0
      },
      "verification": {
        "status": "pending_review"
      }
    }
  ],
  "unresolved_information": [],
  "conflicts_detected": [],
  "market_information": []
}
"""


class GroqLLMExtractor:
    """
    Groq LLM Semantic Extraction Engine.
    Executes raw text -> structured extraction schema JSON conversion.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        self.api_url = settings.groq_api_url
        self.model = model or settings.groq_model

    def extract_from_record(self, record: NormalizedRecord) -> GroqExtractionResult:
        text = record.raw_text_content or str(record.to_dict())
        source_id = record.source_id
        source_type = record.source_type

        user_content = f"Source ID: {source_id}\nSource Type: {source_type}\nTimestamp: {record.timestamp}\nRaw Sales Content:\n{text}"

        print(f"[GROQ EXTRACTION] Request -> Source ID: {source_id} | Length: {len(text)} chars | Model: {self.model}")

        if not self.api_key:
            print("[GROQ EXTRACTION] WARNING -> GROQ_API_KEY is not configured. Running rule-grounded fallback extraction.")
            return self._rule_based_fallback(record)

        try:
            url = self.api_url
            if not url.endswith("/chat/completions"):
                url = f"{url.rstrip('/')}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": GROQ_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
            }

            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                raw_json_str = data["choices"][0]["message"]["content"]
                parsed_json = json.loads(raw_json_str)
                result = GroqExtractionResult(**parsed_json)
                print(f"[GROQ EXTRACTION] Response -> Extracted {len(result.episodes)} episode(s).")
                return result

        except Exception as e:
            print(f"[GROQ EXTRACTION] ERROR -> API Call failed: {str(e)}. Running rule-grounded fallback extraction.")
            return self._rule_based_fallback(record)

    def convert_to_episode(self, ext_ep: ExtractedEpisodeSchema, norm_record: NormalizedRecord) -> Episode:
        """
        Converts Groq ExtractedEpisodeSchema into canonical internal Episode model.
        """
        ep_id = ext_ep.episode_id or f"ep_{uuid.uuid4().hex[:10]}"
        deal_id = ext_ep.deal_id or norm_record.deal_id

        # Extracted components
        situation = ext_ep.context.situation or norm_record.customer_context or "Sales interaction"
        objections = "; ".join(ext_ep.customer.objections) if ext_ep.customer.objections else (
            "; ".join(norm_record.objections) if norm_record.objections else "Unspecified objection"
        )
        tactic = ext_ep.salesperson.tactic or (
            "; ".join(norm_record.salesperson_actions) if norm_record.salesperson_actions else "Standard sales presentation"
        )
        reaction = ext_ep.customer.reaction or (
            "; ".join(norm_record.customer_reactions) if norm_record.customer_reactions else "Reaction not documented"
        )
        outcome_val = ext_ep.outcome.status or norm_record.outcome or "no_decision"
        why_val = ext_ep.outcome.explicit_reason or "Reason not explicitly documented in source"
        lesson_val = ext_ep.learning.lesson or "Document explicit customer reactions and outcomes"

        # Evidence spans and source provenance
        source_ids = ext_ep.provenance.source_ids or [norm_record.source_id]

        return Episode(
            episode_id=ep_id,
            deal_id=deal_id,
            situation=situation,
            objection=objections,
            tactic=tactic,
            customer_reaction=reaction,
            outcome=outcome_val if outcome_val in ["won", "lost", "no_decision"] else "no_decision",
            why=why_val,
            lesson=lesson_val,
            applies_when=ext_ep.learning.applies_when or ["B2B enterprise sales negotiations"],
            did_not_hold_when=ext_ep.learning.did_not_hold_when or ["Budget freeze"],
            pricing_context=ext_ep.commercial.pricing_context or "; ".join(norm_record.pricing_information) or None,
            timestamp=ext_ep.temporal.event_timestamp or norm_record.timestamp,
            valid_as_of=ext_ep.temporal.valid_as_of or norm_record.timestamp[:10],
            source_ids=source_ids,
            extraction_confidence=ext_ep.confidence.extraction_confidence or 0.8,
            causal_confidence=ext_ep.confidence.causal_confidence or 0.4,
            verification_status="pending_review",
        )

    def _rule_based_fallback(self, record: NormalizedRecord) -> GroqExtractionResult:
        """
        Rule-grounded fallback parsing from NormalizedRecord when Groq API is offline/unreachable.
        Never hallucinates missing facts.
        """
        text = record.raw_text_content or ""
        ep_id = f"ep_{uuid.uuid4().hex[:10]}"

        # Check evidence spans
        spans = []
        for line in text.splitlines():
            if any(k in line.lower() for k in ["price", "cost", "quote", "objection", "tactic", "agreed"]):
                spans.append(line.strip()[:100])

        ext_ep = ExtractedEpisodeSchema(
            episode_id=ep_id,
            deal_id=record.deal_id,
            client_id=record.customer_context,
            context={
                "situation": record.customer_context or "Sales deal discussion",
                "pain_points": record.pain_points,
            },
            customer={
                "objections": record.objections,
                "reaction": "; ".join(record.customer_reactions) if record.customer_reactions else None,
            },
            salesperson={
                "tactic": "; ".join(record.salesperson_actions) if record.salesperson_actions else None,
            },
            outcome={
                "status": record.outcome,
                "explicit_reason": None,
            },
            competition=record.competitors,
            commercial={
                "pricing_context": "; ".join(record.pricing_information) if record.pricing_information else None,
            },
            provenance={
                "source_ids": [record.source_id],
                "evidence_spans": spans[:5],
            },
            confidence={
                "extraction_confidence": 0.85 if record.objections else 0.50,
                "causal_confidence": 0.35,
            },
            verification={"status": "pending_review"},
        )

        return GroqExtractionResult(
            extraction_metadata={
                "source_id": record.source_id,
                "source_type": record.source_type,
                "source_timestamp": record.timestamp,
                "extraction_version": "1.0",
                "overall_extraction_confidence": 0.80,
            },
            episodes=[ext_ep],
        )
