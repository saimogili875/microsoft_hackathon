"""
Report Generator module building standard reports across the 5 system report types:
1. Client Relationship Report
2. Live Interaction Report
3. Post-Meeting Change Report
4. Deal Intelligence Report
5. Memory Update Report
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.models.change_event import ChangeEvent, ClientState
from app.models.live_interaction import LiveInteractionState


class ReportGenerator:
    """
    Generates formatted analytical markdown reports supported by empirical pipeline data.
    Guarantees no hallucinated missing information.
    """

    @classmethod
    def generate_client_relationship_report(
        self, client_state: ClientState, historical_records: List[NormalizedRecord]
    ) -> str:
        """
        1. CLIENT RELATIONSHIP REPORT
        For historical / client onboarding analysis.
        """
        stakeholders_str = ", ".join(client_state.stakeholders) if client_state.stakeholders else "None documented"
        tools_str = ", ".join(client_state.tools_used) if client_state.tools_used else "None documented"
        competitors_str = ", ".join(client_state.competitors) if client_state.competitors else "None documented"
        active_obj_str = ", ".join(client_state.active_objections) if client_state.active_objections else "None currently active"
        resolved_obj_str = ", ".join(client_state.resolved_objections) if client_state.resolved_objections else "None documented"
        commitments_str = ", ".join(client_state.commitments) if client_state.commitments else "None documented"

        report = f"""# 📊 CLIENT RELATIONSHIP REPORT
**Deal / Account ID:** `{client_state.deal_id}`
**Customer Context:** {client_state.customer_context}
**Generated Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

---

### 1. Customer Context & Stakeholders
- **Primary Account Context:** {client_state.customer_context}
- **Key Stakeholders:** {stakeholders_str}

### 2. Products & Tools Environment
- **Current Stack / Tools Used:** {tools_str}
- **Known Competitors:** {competitors_str}

### 3. Frictions & Objections
- **Active Objections:** {active_obj_str}
- **Resolved Objections:** {resolved_obj_str}

### 4. Commitments & Open Issues
- **Commitments Made:** {commitments_str}
- **Total Historical Interactions Ingested:** {len(historical_records)}

---
*Note: Fields without empirical source data are left as 'None documented' to prevent hallucinated assumptions.*
"""
        return report.strip()

    @classmethod
    def generate_live_interaction_report(self, live_state: LiveInteractionState) -> str:
        """
        2. LIVE INTERACTION REPORT
        For current calls / meetings in progress.
        """
        signals = live_state.extracted_signals
        concerns_str = "\n".join([f"- {c}" for c in signals.customer_concerns]) or "None identified yet"
        questions_str = "\n".join([f"- {q}" for q in signals.customer_questions]) or "None identified yet"
        tactics_str = "\n".join([f"- {t}" for t in signals.salesperson_tactics]) or "None observed yet"
        reactions_str = "\n".join([f"- {r}" for r in signals.customer_reactions]) or "None observed yet"
        recalls_str = "\n".join([f"- {rec.get('document_id')}: {rec.get('content')[:100]}..." for rec in live_state.recalled_experiences]) or "No Hindsight matches"
        recs_str = "\n".join([f"- {r}" for r in live_state.live_recommendations]) or "Proceed with standard pitch"

        report = f"""# 🎙️ LIVE INTERACTION REPORT
**Session ID:** `{live_state.session_id}`
**Deal ID:** `{live_state.deal_id or 'Unassigned'}`
**Start Time:** {live_state.start_time}

---

### 1. Real-Time Signal Extraction
**Customer Concerns / Objections:**
{concerns_str}

**Customer Questions:**
{questions_str}

**Salesperson Tactics Executed:**
{tactics_str}

**Customer Reactions:**
{reactions_str}

---

### 2. Live Hindsight Memory Recalls
{recalls_str}

---

### 3. Tactical Live Recommendations
{recs_str}
"""
        return report.strip()

    @classmethod
    def generate_post_meeting_change_report(self, changes: List[ChangeEvent]) -> str:
        """
        3. POST-MEETING CHANGE REPORT
        Summarizes detected shifts since previous interaction.
        """
        if not changes:
            return "# 🔄 POST-MEETING CHANGE REPORT\n\n**Status:** No significant state changes detected."

        items = []
        for idx, chg in enumerate(changes, 1):
            items.append(f"""### {idx}. [{chg.category.value.upper()}]
- **Old Information:** {chg.old_information}
- **New Evidence:** {chg.new_evidence}
- **Analytical Shift:** {chg.difference}
- **Status:** `{chg.status}`
""")

        report = f"""# 🔄 POST-MEETING CHANGE REPORT
**Total Changes Detected:** {len(changes)}
**Verification Status:** `PENDING VERIFICATION` (Requires human confirmation before Hindsight retention)

---

{"".join(items)}
"""
        return report.strip()

    @classmethod
    def generate_deal_intelligence_report(
        self,
        client_state: ClientState,
        episodes: List[Episode],
        recalls: List[Dict[str, Any]],
        changes: List[ChangeEvent],
    ) -> str:
        """
        4. DEAL INTELLIGENCE REPORT
        Combines historical experiences + current evidence + risks + unresolved issues.
        """
        top_lessons = "\n".join([f"- **Lesson:** {ep.lesson} (Causal Conf: {ep.causal_confidence})" for ep in episodes[:3]]) or "No verified episodes yet"
        changes_summary = "\n".join([f"- **[{c.category.value}]** {c.difference}" for c in changes[:3]]) or "No recent changes"

        report = f"""# 🧠 DEAL INTELLIGENCE REPORT
**Target Deal:** `{client_state.deal_id}`
**Account:** {client_state.customer_context}

---

### 1. Key Historical Lessons & Tactics
{top_lessons}

### 2. Recent Structural Shifts
{changes_summary}

### 3. Risk & Boundary Factors
- **Active Objections:** {", ".join(client_state.active_objections) or 'None'}
- **Competitors Present:** {", ".join(client_state.competitors) or 'None'}

---
*Generated by Deal Intelligence Agent*
"""
        return report.strip()

    @classmethod
    def generate_memory_update_report(self, episodes: List[Episode]) -> str:
        """
        5. MEMORY UPDATE REPORT
        Details proposed items for Hindsight RETAIN and rationale.
        """
        items = []
        for ep in episodes:
            items.append(f"""### Episode `{ep.episode_id}`
- **Deal ID:** `{ep.deal_id}`
- **Situation:** {ep.situation}
- **Objection:** {ep.objection}
- **Proposed Lesson:** {ep.lesson}
- **Extraction Confidence:** {ep.extraction_confidence}
- **Causal Confidence:** {ep.causal_confidence}
- **Verification Status:** `{ep.verification_status}`
""")

        report = f"""# 💾 HINDSIGHT MEMORY UPDATE REPORT
**Proposed Memory Retentions:** {len(episodes)}

---

{"".join(items)}
"""
        return report.strip()
