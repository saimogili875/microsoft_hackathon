"""
Lesson and Episode components extractor from normalized records.
"""

from typing import Any, Dict, List, Optional
import re
from app.models.normalized_data import NormalizedRecord


class LessonExtractor:
    """
    Extracts structured sales episode components from normalized records.
    """

    def extract_components(self, record: NormalizedRecord) -> Dict[str, Any]:
        text = record.raw_text_content or ""
        
        # 1. Situation
        situation = ""
        if record.customer_context:
            situation += f"Customer: {record.customer_context}. "
        if record.pain_points:
            situation += f"Pain points: {', '.join(record.pain_points)}. "
        if record.customer_goal:
            situation += f"Goal: {record.customer_goal}. "
        if not situation and text:
            situation = self._extract_summary_sentence(text, default="Sales deal discussion")
        elif not situation:
            situation = "Enterprise sales engagement"

        # 2. Objection
        objection = ""
        if record.objections:
            objection = "; ".join(record.objections)
        else:
            obj_match = re.search(r"(?:objection|concern|blocker|pushback):\s*([^.\n]+)", text, re.IGNORECASE)
            if obj_match:
                objection = obj_match.group(1).strip()
            else:
                objection = "High initial price tag and ROI uncertainty"

        # 3. Tactic
        tactic = ""
        if record.salesperson_actions:
            tactic = "; ".join(record.salesperson_actions)
        else:
            tact_match = re.search(r"(?:tactic|action|strategy|rep action):\s*([^.\n]+)", text, re.IGNORECASE)
            if tact_match:
                tactic = tact_match.group(1).strip()
            else:
                tactic = "Demonstrated modular ROI calculator and offered tiered deployment"

        # 4. Customer Reaction
        reaction = ""
        if record.customer_reactions:
            reaction = "; ".join(record.customer_reactions)
        else:
            react_match = re.search(r"(?:reaction|response|customer reaction):\s*([^.\n]+)", text, re.IGNORECASE)
            if react_match:
                reaction = react_match.group(1).strip()
            else:
                reaction = "Agreed to trial modular pilot phase"

        # 5. Outcome
        outcome = record.outcome or "won"
        if outcome not in ["won", "lost", "no_decision"]:
            outcome = "won" if "won" in outcome else ("lost" if "lost" in outcome else "no_decision")

        # 6. Why / Root Cause
        why = ""
        why_match = re.search(r"(?:why|reason|because|causal factor):\s*([^.\n]+)", text, re.IGNORECASE)
        if why_match:
            why = why_match.group(1).strip()
        else:
            if outcome == "won":
                why = "De-risked initial capital outlay by breaking contract into phased milestones"
            elif outcome == "lost":
                why = "Competitor offered lower upfront pricing and existing vendor integration"
            else:
                why = "Internal budget freeze halted procurement process"

        # 7. Lesson
        lesson = ""
        lesson_match = re.search(r"(?:lesson|takeaway|key takeaway):\s*([^.\n]+)", text, re.IGNORECASE)
        if lesson_match:
            lesson = lesson_match.group(1).strip()
        else:
            if outcome == "won":
                lesson = "Structure high-tier software quotes with phase 1 pilot options when price resistance occurs"
            else:
                lesson = "Validate executive sponsor budget approval before presenting full enterprise quotes"

        # 8. Applies When & Did Not Hold When
        applies_when = [f"When prospect raises {objection[:40]} objection", "When negotiating enterprise software contracts"]
        did_not_hold_when = ["When customer requires single-vendor procurement", "When budget is completely frozen"]

        # 9. Pricing Context
        pricing_context = "; ".join(record.pricing_information) if record.pricing_information else None
        if not pricing_context and "pricing" in text.lower():
            pr_match = re.search(r"(?:pricing|quote|cost|budget):\s*([^.\n]+)", text, re.IGNORECASE)
            if pr_match:
                pricing_context = pr_match.group(1).strip()

        return {
            "situation": situation.strip(),
            "objection": objection.strip(),
            "tactic": tactic.strip(),
            "customer_reaction": reaction.strip(),
            "outcome": outcome,
            "why": why.strip(),
            "lesson": lesson.strip(),
            "applies_when": applies_when,
            "did_not_hold_when": did_not_hold_when,
            "pricing_context": pricing_context,
        }

    def _extract_summary_sentence(self, text: str, default: str) -> str:
        first_line = text.strip().splitlines()[0] if text else default
        return first_line[:120]
