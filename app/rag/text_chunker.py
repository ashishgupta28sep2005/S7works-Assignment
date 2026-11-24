"""Speaker-aware, page-local chunking for extracted earnings-call transcripts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from app.ingestion.models import DocumentType, ProcessedDocument
from app.rag.models import TranscriptChunk

_SPEAKER_LINE = re.compile(r"^\s*([A-Z][A-Za-z0-9 .,'’()&/-]{0,65}):(?:\s*)(.*)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def chunk_transcript(
    document: ProcessedDocument,
    *,
    chunk_size: int = 650,
    overlap: int = 100,
    token_counter: Callable[[str], int] | None = None,
) -> list[TranscriptChunk]:
    """Create deterministic chunks while preserving source pages and speaker turns.

    A supplied tokenizer counter gives model-token sizing. If absent, a whitespace
    estimate is used for tests and lightweight inspection before model loading.
    """
    if document.metadata.document_type != DocumentType.EARNINGS_TRANSCRIPT:
        raise ValueError("Financial reports and unknown PDFs cannot be chunked for transcript RAG")
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size >= 1 and 0 <= overlap < chunk_size")
    count_tokens = token_counter or _estimated_token_count
    document_hash = document.metadata.sha256 or hashlib.sha256(
        (document.metadata.original_filename + document.text).encode("utf-8")
    ).hexdigest()
    known_speakers = {name.casefold() for name in document.metadata.management_speakers}
    known_speakers.update({"moderator", "operator"})
    chunks: list[TranscriptChunk] = []
    seen_ids: set[str] = set()
    last_speaker: str | None = None

    for page in document.pages:
        if page.error or not page.text.strip():
            continue
        turns: list[tuple[str | None, str]] = []
        current_speaker = last_speaker
        page_saw_speaker = current_speaker is not None
        current_lines: list[str] = []

        def flush_turn() -> None:
            nonlocal current_lines
            text = "\n".join(current_lines).strip()
            if text and (current_speaker is not None or page_saw_speaker):
                turns.append((current_speaker, text))
            current_lines = []

        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue
            match = _SPEAKER_LINE.match(line)
            label = match.group(1).strip() if match else None
            # Transcript analyst turns also matter for context. Keep them separate
            # from management turns so downstream analysis cannot treat a question
            # as management guidance.
            if label and (label.casefold() in known_speakers or len(label.split()) >= 2):
                flush_turn()
                current_speaker = label
                page_saw_speaker = True
                remainder = match.group(2).strip()
                if remainder:
                    current_lines.append(remainder)
            else:
                current_lines.append(line)
        flush_turn()
        last_speaker = current_speaker
        if not turns:
            continue

        units: list[tuple[str | None, str]] = []
        for speaker, body in turns:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            for paragraph in paragraphs:
                for sentence in _SENTENCE_SPLIT.split(paragraph):
                    sentence = sentence.strip()
                    if sentence:
                        rendered = f"{speaker}: {sentence}" if speaker else sentence
                        units.extend((speaker, part) for part in _split_oversized(
                            rendered, chunk_size, count_tokens
                        ))

        page_units: list[tuple[str | None, str]] = []
        current: list[tuple[str | None, str]] = []
        current_size = 0

        def emit() -> None:
            nonlocal current, current_size
            text = "\n\n".join(part for _, part in current).strip()
            if text:
                speakers = {speaker for speaker, _ in current if speaker}
                speaker = next(iter(speakers)) if len(speakers) == 1 else None
                roles = {_speaker_role(name, known_speakers) for name, _ in current}
                roles.discard(None)
                speaker_role = next(iter(roles)) if len(roles) == 1 else None
                chunk_id = hashlib.sha256(
                    f"{document_hash}:{page.page_number}:{text}".encode("utf-8")
                ).hexdigest()
                if chunk_id not in seen_ids:
                    chunks.append(TranscriptChunk(
                        company=document.metadata.company or "TCS",
                        document_name=document.metadata.original_filename,
                        document_type=DocumentType.EARNINGS_TRANSCRIPT.value,
                        reporting_period=document.metadata.reporting_period,
                        earnings_call_date=document.metadata.earnings_call_date,
                        page_number=page.page_number,
                        speaker=speaker,
                        speaker_role=speaker_role,
                        chunk_id=chunk_id,
                        original_text=text,
                        document_hash=document_hash,
                    ))
                    seen_ids.add(chunk_id)
            current = []
            current_size = 0

        for unit in units:
            unit_size = count_tokens(unit[1])
            if current and _speaker_role(current[-1][0], known_speakers) != _speaker_role(unit[0], known_speakers):
                page_units = []
                emit()
            if current and current_size + unit_size > chunk_size:
                page_units.extend(current)
                emit()
                carry: list[tuple[str | None, str]] = []
                carry_size = 0
                for previous in reversed(page_units):
                    size = count_tokens(previous[1])
                    if carry and carry_size + size > overlap:
                        break
                    if not carry and size > overlap:
                        break
                    carry.insert(0, previous)
                    carry_size += size
                current = carry
                current_size = carry_size
            current.append(unit)
            current_size += unit_size
        if current:
            emit()

    return chunks


def _estimated_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text))) if text.strip() else 0


def _split_oversized(text: str, max_tokens: int, counter: Callable[[str], int]) -> list[str]:
    if counter(text) <= max_tokens:
        return [text]
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join((*current, word))
        if current and counter(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _speaker_role(speaker: str | None, management_speakers: set[str]) -> str | None:
    if not speaker:
        return None
    normalized = speaker.casefold()
    if normalized in management_speakers:
        return "management"
    if normalized in {"moderator", "operator", "conference operator"}:
        return "moderator"
    return "analyst"
