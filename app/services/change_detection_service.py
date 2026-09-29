"""
Change Detection Service for Mode 02-B — Post-Meeting Change Detection.
Compares previous deal/client state with new interaction and detects shifts across 18 change categories.
"""

from typing import Any, Dict, List, Optional
import re
from app.models.change_event import ChangeCategory, ChangeEvent, ClientState
from app.models.normalized_data import NormalizedRecord


class ChangeDetectionService:
    """
    Engine comparing baseline ClientState against a new NormalizedRecord interaction.
    Outputs ChangeEvent objects in 'pending_review' state without destroying historical memories.
    """

    def detect_changes(self, baseline: ClientState, new_interaction: NormalizedRecord) -> List[ChangeEvent]:
        changes: List[ChangeEvent] = []
        source_ids = [ref.get("raw_id", "") for ref in new_interaction.source_references if ref.get("raw_id")]
        deal_id = new_interaction.deal_id or baseline.deal_id
        text = (new_interaction.raw_text_content or "").lower()

        # 1. New Competitor
        for comp in new_interaction.competitors:
            if comp and comp.lower() not in [c.lower() for c in baseline.competitors]:
                changes.append(ChangeEvent(
                    deal_id=deal_id,
                    category=ChangeCategory.NEW_COMPETITOR,
                    old_information=f"Known competitors: {', '.join(baseline.competitors) or 'None'}",
                    new_evidence=f"Mentioned new competitor: {comp}",
                    difference=f"Market competition expanded to include {comp}.",
                    source_ids=source_ids,
                ))

        # 2. Competitor Price Change
        for pricing in new_interaction.pricing_information:
            for old_comp, old_price in baseline.competitor_pricing.items():
                if old_comp.lower() in pricing.lower():
                    changes.append(ChangeEvent(
                        deal_id=deal_id,
                        category=ChangeCategory.COMPETITOR_PRICE_CHANGE,
                        old_information=f"{old_comp} pricing baseline ≈ {old_price}",
                        new_evidence=f"New report: {pricing}",
                        difference=f"{old_comp} price adjustment detected.",
                        source_ids=source_ids,
                    ))

        # 3. Customer Switching Tools
        if "switching from" in text or "migrating away from" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.CUSTOMER_SWITCHING_TOOLS,
                old_information=f"Tools used: {', '.join(baseline.tools_used) or 'Existing stack'}",
                new_evidence="Customer reports active migration from legacy tool stack.",
                difference="Tool migration creates window for platform adoption.",
                source_ids=source_ids,
            ))

        # 4. Customer Adopting Tool
        if "adopting" in text or "implementing" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.CUSTOMER_ADOPTING_TOOL,
                old_information="Stack baseline",
                new_evidence="Customer adopting new complementary tool.",
                difference="Integration requirement shift.",
                source_ids=source_ids,
            ))

        # 5. Customer Using Product Differently
        if "new use case" in text or "using it for" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.CUSTOMER_USING_PRODUCT_DIFFERENTLY,
                old_information="Standard usage pattern",
                new_evidence="Customer expanding product to secondary department use case.",
                difference="Expansion opportunity identified.",
                source_ids=source_ids,
            ))

        # 6. New Stakeholder
        for sh in new_interaction.stakeholders:
            if sh and sh.lower() not in [s.lower() for s in baseline.stakeholders]:
                changes.append(ChangeEvent(
                    deal_id=deal_id,
                    category=ChangeCategory.NEW_STAKEHOLDER,
                    old_information=f"Known stakeholders: {', '.join(baseline.stakeholders) or 'None'}",
                    new_evidence=f"New participant/decision maker: {sh}",
                    difference=f"Expanded buying committee with {sh}.",
                    source_ids=source_ids,
                ))

        # 7. Stakeholder Leaving
        if "left company" in text or "resigned" in text or "no longer at" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.STAKEHOLDER_LEAVING,
                old_information=f"Stakeholders: {', '.join(baseline.stakeholders)}",
                new_evidence="Key sponsor or champion left the organization.",
                difference="High deal risk: Champion departure requires re-engaging new leadership.",
                source_ids=source_ids,
            ))

        # 8. New Customer Requirement
        if "requirement" in text or "must have" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.NEW_CUSTOMER_REQUIREMENT,
                old_information="Baseline requirements",
                new_evidence="New mandatory technical or compliance requirement specified.",
                difference="Evaluation criteria scope expanded.",
                source_ids=source_ids,
            ))

        # 9. New Objection
        for obj in new_interaction.objections:
            if obj and obj.lower() not in [o.lower() for o in baseline.active_objections]:
                changes.append(ChangeEvent(
                    deal_id=deal_id,
                    category=ChangeCategory.NEW_OBJECTION,
                    old_information=f"Active objections: {', '.join(baseline.active_objections) or 'None'}",
                    new_evidence=f"New objection raised: {obj}",
                    difference=f"Fresh friction point encountered: {obj}.",
                    source_ids=source_ids,
                ))

        # 10. Resolved Objection
        if "objection resolved" in text or "addressed concern" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.RESOLVED_OBJECTION,
                old_information=f"Active objections: {', '.join(baseline.active_objections)}",
                new_evidence="Customer confirmed previous concern is resolved.",
                difference="Friction cleared; deal progressing.",
                source_ids=source_ids,
            ))

        # 11. New Pricing Expectation
        if "budget limit" in text or "target price" in text or "expectation" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.NEW_PRICING_EXPECTATION,
                old_information="Previous quote baseline",
                new_evidence="Customer specified revised target pricing expectation.",
                difference="Negotiation envelope shifted.",
                source_ids=source_ids,
            ))

        # 12. New Business Priority
        if "priority" in text or "quarterly goal" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.NEW_BUSINESS_PRIORITY,
                old_information="Previous strategic focus",
                new_evidence="Executive team reprioritized business objectives.",
                difference="Value proposition must align with new executive priorities.",
                source_ids=source_ids,
            ))

        # 13. New Company / Workflow Change
        if "reorganization" in text or "reorg" in text or "workflow change" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.NEW_COMPANY_WORKFLOW_CHANGE,
                old_information="Legacy organizational structure",
                new_evidence="Internal reorganization underway.",
                difference="Process and approvals restructured.",
                source_ids=source_ids,
            ))

        # 14. New Pain Point
        for pp in new_interaction.pain_points:
            if pp and pp.lower() not in [p.lower() for p in baseline.pain_points]:
                changes.append(ChangeEvent(
                    deal_id=deal_id,
                    category=ChangeCategory.NEW_PAIN_POINT,
                    old_information=f"Pain points: {', '.join(baseline.pain_points) or 'None'}",
                    new_evidence=f"New pain point identified: {pp}",
                    difference=f"Additional operational challenge: {pp}.",
                    source_ids=source_ids,
                ))

        # 15. New Commitment
        if "agreed to" in text or "committed to" in text or "next step" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.NEW_COMMITMENT,
                old_information=f"Previous commitments: {', '.join(baseline.commitments) or 'None'}",
                new_evidence="Customer committed to upcoming milestone/action.",
                difference="Deal velocity increasing.",
                source_ids=source_ids,
            ))

        # 16. Previous Commitment Invalid
        if "canceled meeting" in text or "pushed back" in text or "missed deadline" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.PREVIOUS_COMMITMENT_INVALID,
                old_information=f"Commitments: {', '.join(baseline.commitments)}",
                new_evidence="Customer missed or canceled previously agreed milestone.",
                difference="Slippage risk: Schedule invalidated.",
                source_ids=source_ids,
            ))

        # 17. New Market Information
        if "market trend" in text or "industry shift" in text or "regulatory" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.NEW_MARKET_INFORMATION,
                old_information="Market context",
                new_evidence="Macro market or industry change noted in discussion.",
                difference="External market context shift.",
                source_ids=source_ids,
            ))

        # 18. Customer Relationship Change
        if new_interaction.outcome in ["won", "lost"] or "partnership" in text:
            changes.append(ChangeEvent(
                deal_id=deal_id,
                category=ChangeCategory.CUSTOMER_RELATIONSHIP_CHANGE,
                old_information="Prospect status",
                new_evidence=f"Deal status transitioned to {new_interaction.outcome or 'partner'}.",
                difference=f"Account status changed to {new_interaction.outcome}.",
                source_ids=source_ids,
            ))

        return changes
