"""
Main Entry Point for Deal Intelligence Agent Data Pipeline CLI.
Supports Mode 01 Live Calls, Mode 02 Historical Ingestion, Change Detection, and Report Generation.
"""

import sys
import argparse
import json
from pathlib import Path
from app.services.pipeline import DealIntelligencePipeline
from app.models.live_interaction import LiveTranscriptChunk
from app.models.change_event import ClientState


def main():
    parser = argparse.ArgumentParser(description="Deal Intelligence Data Pipeline CLI v2.0")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest subcommand (Mode 02)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a historical dataset file (Mode 02)")
    ingest_parser.add_argument("--file", required=True, help="Path to input file (.json, .csv, .txt, etc.)")
    ingest_parser.add_argument("--type", default="auto", help="Source type label (default: auto)")

    # Live Call subcommand (Mode 01)
    live_parser = subparsers.add_parser("live", help="Simulate or send a live call chunk (Mode 01)")
    live_parser.add_argument("--session-id", required=True, help="Call session identifier")
    live_parser.add_argument("--speaker", default="Customer", help="Speaker name (Rep/Customer)")
    live_parser.add_argument("--text", required=True, help="Transcript text snippet")
    live_parser.add_argument("--finalize", action="store_true", help="Finalize call session post-meeting")

    # Review subcommand
    review_parser = subparsers.add_parser("review", help="Review pending episodes")
    review_parser.add_argument("--list", action="store_true", help="List all pending episodes")
    review_parser.add_argument("--episode-id", help="Target episode ID")
    review_parser.add_argument("--action", choices=["confirm", "correct", "reject"], help="Verification action")
    review_parser.add_argument("--role", default="account_executive", help="Reviewer role")

    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate pipeline reports")
    report_parser.add_argument("--type", choices=["client_relationship", "live_interaction", "post_meeting_change", "deal_intelligence", "memory_update"], required=True, help="Report type")

    # Trace subcommand
    trace_parser = subparsers.add_parser("trace", help="Trace lineage of a memory or episode")
    trace_parser.add_argument("--episode-id", required=True, help="Target episode ID")

    # Recall subcommand
    recall_parser = subparsers.add_parser("recall", help="Recall retained memories from Hindsight")
    recall_parser.add_argument("--query", required=True, help="Search query string")

    args = parser.parse_args()
    pipeline = DealIntelligencePipeline()

    if args.command == "ingest":
        print(f"Processing file: {args.file}")
        res = pipeline.process_file(args.file, source_type=args.type)
        print("Ingestion Result:")
        print(json.dumps(res, indent=2))

    elif args.command == "live":
        chunk = LiveTranscriptChunk(
            session_id=args.session_id,
            speaker=args.speaker,
            text=args.text,
        )
        res = pipeline.process_live_chunk(chunk)
        print("Live Chunk Signal Extraction & Hindsight Recall:")
        print(json.dumps(res, indent=2))

        if args.finalize:
            fin = pipeline.finalize_live_session(args.session_id)
            print("\nPost-Meeting Episode Candidate Generated:")
            print(fin["episode_candidate"].model_dump())

    elif args.command == "review":
        if args.list:
            pending = pipeline.verification_service.get_pending_episodes()
            print(f"Pending Review Episodes ({len(pending)}):")
            for ep in pending:
                print(f"  - [{ep.episode_id}] Deal: {ep.deal_id} | Situation: {ep.situation[:60]}... | Status: {ep.verification_status}")
        elif args.episode_id and args.action:
            res = pipeline.verify_and_retain(args.episode_id, action=args.action, role=args.role)
            print("Verification Result:")
            print(json.dumps(res, indent=2))

    elif args.command == "report":
        # Demo report generation with default context
        state = ClientState(
            deal_id="D-DEMO-001",
            customer_context="Enterprise Logistics Inc",
            stakeholders=["John Doe (VP Ops)", "Jane Smith (CTO)"],
            tools_used=["Salesforce", "Legacy ERP"],
            competitors=["Competitor X"],
            active_objections=["Implementation cost too high"],
        )
        pending_episodes = pipeline.verification_service.get_pending_episodes()

        context = {
            "client_state": state,
            "episodes": pending_episodes,
            "historical_records": [],
            "recalls": [],
            "changes": [],
        }
        report = pipeline.generate_report(args.type, context)
        print(report)

    elif args.command == "trace":
        res = pipeline.traceability_service.trace_by_episode_id(args.episode_id)
        print("Traceability Chain:")
        print(json.dumps(res, indent=2))

    elif args.command == "recall":
        res = pipeline.hindsight_client.recall(args.query)
        print("Matching Hindsight Memories:")
        for idx, item in enumerate(res, 1):
            print(f"\n--- [{idx}] Score: {item['score']} | Doc ID: {item['document_id']} ---")
            print(item["content"])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
