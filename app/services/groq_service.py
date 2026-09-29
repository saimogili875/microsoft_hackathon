"""
Groq Service handling LLM reasoning, memory comparison, manual chat, and report generation.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)


class GroqService:
    """
    Groq LLM Reasoning Client.
    Responsible for reasoning over recalled Hindsight memories, comparing current vs historical information,
    answering user chat queries, and formatting intelligence reports.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.groq_api_key
        self.api_url = api_url or settings.groq_api_url
        self.model = model or settings.groq_model

    def generate_response(self, context_payload: Dict[str, Any]) -> str:
        user_query = context_payload.get("user_query", "Analyze deal intelligence data.")
        hindsight_memories = context_payload.get("hindsight_memories", [])
        task = context_payload.get("task", "Generate deal intelligence response")

        # Logging tag (NEVER log secrets!)
        print(f"[GROQ] REQUEST -> Task: {task} | Query: '{user_query}' | Memories: {len(hindsight_memories)}")

        system_prompt = (
            "You are the Deal Intelligence Agent AI Assistant. "
            "Your task is to analyze sales data using strictly provided historical Hindsight memories, "
            "raw source excerpts, and detected changes. "
            "CRITICAL RULES:\n"
            "1. Base answers ONLY on provided empirical evidence.\n"
            "2. If Hindsight memories are empty, explicitly state that no relevant historical memories were found.\n"
            "3. NEVER invent or hallucinate missing facts, pricing, or outcomes.\n"
            "4. Clearly distinguish between verified historical facts and pending unverified evidence.\n"
            "5. If conflicting competitor pricing or evidence exists, point out the discrepancy clearly."
        )

        user_content = json.dumps(context_payload, indent=2, default=str)

        if not self.api_key:
            print("[GROQ] WARNING -> GROQ_API_KEY is not configured. Returning deterministic analytical response.")
            return self._fallback_reasoning(context_payload)

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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Structured Context Payload:\n{user_content}"},
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
            }

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"[GROQ] RESPONSE -> Successfully generated {len(content)} chars.")
                return content

        except Exception as e:
            print(f"[GROQ] ERROR -> Remote API call unavailable ({str(e)}). Executing safe local reasoning fallback.")
            return self._fallback_reasoning(context_payload)

    def _fallback_reasoning(self, payload: Dict[str, Any]) -> str:
        user_query = payload.get("user_query", "Deal Analysis")
        hindsight_memories = payload.get("hindsight_memories", [])
        detected_changes = payload.get("detected_changes", [])

        out = []
        out.append(f"### DEAL INTELLIGENCE RESPONSE")
        out.append(f"**Query/Task:** {user_query}")

        if hindsight_memories:
            out.append("\n#### 🧠 Recalled Hindsight Historical Memories:")
            for idx, mem in enumerate(hindsight_memories, 1):
                doc_id = mem.get("document_id", f"mem_{idx}")
                content = mem.get("content", str(mem))
                out.append(f"{idx}. **[{doc_id}]**\n{content}\n")
        else:
            out.append("\n#### 🧠 Recalled Hindsight Historical Memories:")
            out.append("*No relevant historical memories were found in Hindsight for this query.*\n")

        if detected_changes:
            out.append("#### 🔄 Detected State Shifts & Conflicts:")
            for chg in detected_changes:
                cat = chg.get("category", "shift")
                old_info = chg.get("old_information", "")
                new_info = chg.get("new_evidence", "")
                diff = chg.get("difference", "")
                out.append(f"- **[{cat}]** Old: '{old_info}' | New: '{new_info}' → {diff}")
            out.append("")

        out.append("---")
        out.append("*Response generated based on empirical source evidence.*")
        return "\n".join(out)
