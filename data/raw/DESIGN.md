# DEAL INTELLIGENCE AGENT - DESIGN SPECIFICATION

## System Core Principles
1. **Common Hindsight Contract**: All data modes (Mode 01 live streams and Mode 02 historical files) normalize into identical canonical Episode records.
2. **Hindsight Memory Layer**: Handles persistent retention of verified experiences and relevant retrieval (RECALL).
3. **Groq Reasoning Layer**: Performs contextual reasoning, memory comparison, and report generation using structured prompt context.
4. **Data Traceability**: Raw data remains stored on disk as the source of truth. Every retained Hindsight memory links back to raw source IDs.
5. **No Hallucination**: Unsubstantiated claims are marked as `unknown`, `pending_verification`, or `not available`.
