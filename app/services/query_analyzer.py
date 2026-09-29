"""
Query Analyzer module for ARCH-2 Query Understanding.
Transforms raw salesperson query strings into structured RecallQuery objects.
"""

from typing import Any, Dict, Optional
import re
from app.models.recall_query import RecallQuery


class QueryAnalyzer:
    """
    Query-understanding engine for ARCH-2.
    Determines client, deal, stage, topic, objection, competitor, and requested information
    without hallucinating values (uses None when unavailable).
    """

    KNOWN_CLIENTS = [
        "ABC Logistics", "Global Cargo Corp", "Enterprise Tech",
        "Acme Corp", "Delta Logistics", "Apex Corp", "Global Logistics"
    ]

    KNOWN_COMPETITORS = [
        "Competitor X", "Competitor Y", "Competitor Z",
        "Acme", "Salesforce", "HubSpot", "Oracle"
    ]

    def analyze_query(
        self,
        query: str,
        client_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        stage: Optional[str] = None,
        current_context: Optional[Dict[str, Any]] = None,
    ) -> RecallQuery:
        q_lower = query.lower()

        # 1. Identify Client
        extracted_client = client_id
        if not extracted_client:
            for client in self.KNOWN_CLIENTS:
                if client.lower() in q_lower:
                    extracted_client = client
                    break

        # 2. Identify Competitor
        extracted_competitor = None
        for comp in self.KNOWN_COMPETITORS:
            if comp.lower() in q_lower:
                extracted_competitor = comp
                break

        # 3. Identify Objection / Topic
        extracted_objection = None
        extracted_topic = None

        if any(w in q_lower for w in ["price", "cost", "pricing", "budget", "quote", "discount"]):
            extracted_topic = "pricing"
            if "implementation cost" in q_lower or "implementation price" in q_lower:
                extracted_objection = "implementation cost"
            elif "price" in q_lower or "cost" in q_lower:
                extracted_objection = "pricing"

        elif "competitor" in q_lower or "competition" in q_lower:
            extracted_topic = "competitor"
        elif "implementation" in q_lower or "timeline" in q_lower:
            extracted_topic = "implementation"
            extracted_objection = "implementation timeline"
        elif "compliance" in q_lower or "security" in q_lower or "sovereignty" in q_lower:
            extracted_topic = "compliance"
            extracted_objection = "data sovereignty compliance"

        # 4. Identify Requested Information
        requested_info = []
        if any(w in q_lower for w in ["worked", "tactic", "strategy", "action", "do"]):
            requested_info.append("tactic")
            requested_info.append("reaction")
            requested_info.append("outcome")
        if any(w in q_lower for w in ["happened", "history", "previous", "before"]):
            requested_info.append("sales_experience")
            requested_info.append("outcomes")
        if any(w in q_lower for w in ["competitor", "encountered"]):
            requested_info.append("competitors")
        if any(w in q_lower for w in ["price", "cost", "quote"]):
            requested_info.append("pricing_context")

        if not requested_info:
            requested_info = ["tactic", "reaction", "outcome", "lesson"]

        # Scope Strategy
        strategy = "semantic_narrow"
        if extracted_client and extracted_objection:
            strategy = "client_scoped"
        elif extracted_client:
            strategy = "client_scoped"
        elif extracted_objection or extracted_competitor:
            strategy = "topic_scoped"
        else:
            strategy = "broad"

        return RecallQuery(
            original_query=query,
            client_id=extracted_client,
            deal_id=deal_id,
            stage=stage,
            topic=extracted_topic,
            objection=extracted_objection,
            competitor=extracted_competitor,
            requested_information=list(set(requested_info)),
            retrieval_strategy=strategy,
        )
