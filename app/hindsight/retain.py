"""
Retain payload formatter and retainer helper for Hindsight Memory.
"""

from typing import Any, Dict
from app.models.episode import Episode
from app.models.market_info import MarketInfoRecord
from app.models.memory import HindsightMemoryPayload
from app.hindsight.client import HindsightClient


def format_episode_narrative(episode: Episode) -> str:
    """
    Formats a confirmed sales episode into concise self-contained narrative text block
    matching exact required Hindsight format.
    """
    applies_str = ", ".join(episode.applies_when) if episode.applies_when else "General sales negotiations"
    did_not_str = ", ".join(episode.did_not_hold_when) if episode.did_not_hold_when else "None identified"
    pricing_str = episode.pricing_context or "Not specified"
    verified_str = f"{episode.verified_by_role or 'account_executive'} ({episode.verification_status})"
    valid_str = episode.valid_as_of or episode.timestamp[:10]

    narrative = f"""[EPISODE]
Situation:
{episode.situation}

Objection:
{episode.objection}

Tactic:
{episode.tactic}

Customer reaction:
{episode.customer_reaction}

Outcome:
{episode.outcome}

Why:
{episode.why}

Applies when:
{applies_str}

Did not hold when:
{did_not_str}

Lesson:
{episode.lesson}

Pricing context:
{pricing_str}

Verified by:
{verified_str}

Valid as of:
{valid_str}"""
    return narrative.strip()


def retain_episode(episode: Episode, client: HindsightClient) -> HindsightMemoryPayload:
    """
    Converts a confirmed episode into a Hindsight retain payload and sends it to Hindsight.
    Uses stable document ID 'episode:<episode_id>'.
    """
    doc_id = f"episode:{episode.episode_id}"
    narrative = format_episode_narrative(episode)
    
    metadata = {
        "episode_id": episode.episode_id,
        "deal_id": episode.deal_id,
        "source_ids": episode.source_ids,
        "outcome": episode.outcome,
        "extraction_confidence": episode.extraction_confidence,
        "causal_confidence": episode.causal_confidence,
        "verification_status": episode.verification_status,
        "verified_by_role": episode.verified_by_role,
        "verified_at": episode.verified_at,
        "valid_as_of": episode.valid_as_of,
    }

    payload = HindsightMemoryPayload(
        document_id=doc_id,
        content=narrative,
        metadata=metadata,
    )
    client.retain(payload)
    return payload


def format_market_info_narrative(record: MarketInfoRecord) -> str:
    pricing_str = record.pricing_context or "Standard market rates"
    verified_str = f"{record.verified_by_role or 'market_analyst'} ({record.status})"
    valid_until_str = record.valid_until or "Present"

    narrative = f"""[MARKET_FACT]
Competitor:
{record.competitor}

Topic:
{record.topic}

Statement:
{record.statement}

Pricing context:
{pricing_str}

Status:
{record.status}

Valid from:
{record.valid_from}

Valid until:
{valid_until_str}

Verified by:
{verified_str}"""
    return narrative.strip()


def retain_market_info(record: MarketInfoRecord, client: HindsightClient) -> HindsightMemoryPayload:
    """
    Converts a market info record into Hindsight retain payload.
    Uses stable document ID 'market:<competitor>:<date>'.
    """
    comp_clean = record.competitor.lower().replace(" ", "_")
    doc_id = f"market:{comp_clean}:{record.valid_from}"
    narrative = format_market_info_narrative(record)

    metadata = {
        "market_id": record.market_id,
        "competitor": record.competitor,
        "topic": record.topic,
        "source_ids": record.source_ids,
        "status": record.status,
        "valid_from": record.valid_from,
        "valid_until": record.valid_until,
    }

    payload = HindsightMemoryPayload(
        document_id=doc_id,
        content=narrative,
        metadata=metadata,
    )
    client.retain(payload)
    return payload
