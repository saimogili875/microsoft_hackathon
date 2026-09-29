"""
Field Mapper for mapping arbitrary raw key names to canonical normalized schema fields.
"""

from typing import Any, Dict, List, Optional


class FieldMapper:
    """
    Maps heterogenous keys from various sales tools (HubSpot, Salesforce, text files, custom JSONs)
    to standardized internal fields.
    """

    KEY_ALIASES = {
        "deal_id": ["deal_id", "deal_uuid", "opportunity_id", "opp_id", "dealid", "crm_deal_id"],
        "customer_context": ["customer_context", "client_name", "account_name", "company", "customer", "prospect", "client"],
        "deal_stage": ["deal_stage", "stage", "pipeline_stage", "status", "phase"],
        "stakeholders": ["stakeholders", "buyers", "contacts", "decision_makers", "participants", "attendees"],
        "customer_goal": ["customer_goal", "objective", "target_goal", "business_goal", "desired_outcome"],
        "pain_points": ["pain_points", "challenges", "problems", "painpoints", "issues", "frustrations"],
        "objections": ["objections", "pushback", "concerns", "blockers", "hesitations", "customer_objection"],
        "competitors": ["competitors", "competition", "rivals", "alternative_vendors"],
        "pricing_information": ["pricing_information", "pricing", "quote", "budget", "cost", "deal_size", "price", "discount"],
        "salesperson_actions": ["salesperson_actions", "actions", "tactics", "strategy", "rep_action", "next_steps"],
        "customer_reactions": ["customer_reactions", "reaction", "customer_feedback", "client_response"],
        "outcome": ["outcome", "result", "deal_result", "won_lost", "close_reason"],
    }

    @classmethod
    def find_key(cls, raw_dict: Dict[str, Any], canonical_field: str) -> Optional[Any]:
        if not isinstance(raw_dict, dict):
            return None

        # Check exact key first
        if canonical_field in raw_dict and raw_dict[canonical_field] is not None:
            return raw_dict[canonical_field]

        # Check aliases
        aliases = cls.KEY_ALIASES.get(canonical_field, [])
        for alias in aliases:
            for k, v in raw_dict.items():
                if k.lower().strip() == alias and v is not None:
                    return v
        return None
