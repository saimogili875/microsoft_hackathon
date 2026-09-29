"""
Email Loader for ingestion of EML files and email text/JSON payloads.
Extracts sender, recipients, subject, timestamp, body, and attachment metadata.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import email
from email.policy import default
import re
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector


class EmailLoader(BaseLoader):
    """
    Loader for email documents (.eml, email JSON, email text).
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        source_id = f"email_{FileDetector.calculate_hash(path)[:12]}"
        stype = "email" if source_type == "auto" else source_type

        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=default)

        sender = msg.get("From", "unknown")
        recipients = [r.strip() for r in msg.get("To", "").split(",") if r.strip()]
        subject = msg.get("Subject", "No Subject")
        date_hdr = msg.get("Date")

        body_parts = []
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition"))
                filename = part.get_filename()
                if filename:
                    attachments.append({
                        "file_name": filename,
                        "content_type": part.get_content_type(),
                        "size_bytes": len(part.get_payload(decode=True) or b""),
                    })
                elif "attachment" in content_disposition:
                    attachments.append({"content_type": part.get_content_type()})
                else:
                    part_text = part.get_content()
                    if isinstance(part_text, str) and part.get_content_type() in ["text/plain", "text/html"]:
                        body_parts.append(part_text)
        else:
            body_parts.append(msg.get_content() if isinstance(msg.get_content(), str) else str(msg.get_content()))

        body_text = "\n\n".join(body_parts)

        deal_id = self._extract_field(body_text, r"(?:deal_id|deal):\s*([A-Za-z0-9_\-]+)")
        customer_id = self._extract_field(body_text, r"(?:customer_id|client_id|client):\s*([A-Za-z0-9_\-\s]+)")
        salesperson_id = self._extract_field(body_text, r"(?:salesperson_id|rep):\s*([A-Za-z0-9_\-\s]+)")

        metadata = {
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "attachments": attachments,
            "attachment_count": len(attachments),
            "date_header": date_hdr,
        }

        content_payload = {
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "body": body_text,
            "attachments": attachments,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=date_hdr or datetime.now(timezone.utc).isoformat(),
            file_format="eml",
            raw_content=content_payload,
            metadata=metadata,
            deal_id=deal_id,
            customer_id=customer_id,
            salesperson_id=salesperson_id,
            source_location=str(path.resolve()),
            extraction_method="email_parse",
            extraction_confidence=1.0,
        )

        return [record]

    def load_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawRecord]:
        meta = metadata or {}
        if isinstance(content, dict):
            sender = content.get("sender") or content.get("from") or meta.get("sender")
            recipients = content.get("recipients") or content.get("to") or meta.get("recipients", [])
            subject = content.get("subject") or meta.get("subject", "Email Interaction")
            body = content.get("body") or content.get("text") or str(content)
            attachments = content.get("attachments") or meta.get("attachments", [])
        else:
            sender = meta.get("sender")
            recipients = meta.get("recipients", [])
            subject = meta.get("subject", "Email Interaction")
            body = str(content)
            attachments = meta.get("attachments", [])

        payload = {
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "body": body,
            "attachments": attachments,
        }

        full_meta = {
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "attachments": attachments,
            **meta,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=source_type or "email",
            file_name=meta.get("file_name"),
            timestamp=meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format="eml",
            raw_content=payload,
            metadata=full_meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="email_parse",
            extraction_confidence=1.0,
        )
        return [record]

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
