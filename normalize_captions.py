#!/usr/bin/env python3
"""
normalize_captions.py

Generic caption cleanup pipeline for Whisper.cpp TXT/VTT output.

Purpose:
- Convert Whisper-style TXT or rough VTT into valid WebVTT.
- Improve caption readability and accessibility.
- Apply conservative dictionary-based terminology corrections.
- Produce notes so edits are reviewable.

This is intentionally generic. It does not know about a specific event agenda.
Use layered dictionaries for stable terminology only, for example:

  dictionaries/un-core.yml
  dictionaries/un-agencies.yml
  dictionaries/un-digital.yml
  dictionaries/geography.yml

Example:

  python normalize_captions.py transcripts/*.txt \
    --dictionary dictionaries/un-core.yml \
    --dictionary dictionaries/un-agencies.yml \
    --dictionary dictionaries/un-digital.yml \
    --out-dir captions

Input supported:
- Whisper.cpp TXT lines like:
  [00:00:00.000 --> 00:00:03.500] Hello.
- WebVTT files beginning with WEBVTT

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


def ts_to_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
    elif len(parts) == 2:
        h = "0"
        m, rest = parts
    else:
        raise ValueError(f"Bad timestamp: {ts!r}")

    if "." in rest:
        s, ms = rest.split(".", 1)
    else:
        s, ms = rest, "0"

    ms = (ms + "000")[:3]
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def ms_to_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, rem = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{rem:03d}"


def parse_input(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if raw.lstrip().startswith("WEBVTT"):
        return parse_vtt(raw)
    return parse_whisper_txt(raw)


def parse_whisper_txt(raw: str) -> list[Cue]:
    cues: list[Cue] = []
    pattern = re.compile(
        r"^\[(\d\d:\d\d:\d\d[\.,]\d{3})\s+-->\s+(\d\d:\d\d:\d\d[\.,]\d{3})\]\s*(.*)$"
    )
    for line in raw.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        start, end, text = m.groups()
        try:
            cues.append(Cue(ts_to_ms(start), ts_to_ms(end), text.strip()))
        except ValueError:
            continue
    return cues


def parse_vtt(raw: str) -> list[Cue]:
    cues: list[Cue] = []
    lines = raw.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "REGION")):
            i += 1
            continue

        # Optional cue identifier.
        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" in line:
            try:
                start_part, end_part = line.split("-->", 1)
                start = start_part.strip().split()[0]
                end = end_part.strip().split()[0]
                i += 1
                body: list[str] = []
                while i < len(lines) and lines[i].strip():
                    body.append(lines[i].strip())
                    i += 1
                cues.append(Cue(ts_to_ms(start), ts_to_ms(end), " ".join(body).strip()))
            except Exception:
                i += 1
        else:
            i += 1

    return cues


def load_dictionary(path: Path) -> dict[str, list[str]]:
    """
    Load simple YAML or JSON dictionaries.

    YAML format supported:

      organizations:
        - United Nations
        - UNDP

    JSON format supported:

      {"organizations": ["United Nations", "UNDP"]}
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        return {
            str(k): [str(x) for x in v]
            for k, v in data.items()
            if isinstance(v, list)
        }

    out: dict[str, list[str]] = {}
    current: str | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith((" ", "\t")) and stripped.endswith(":"):
            current = stripped[:-1]
            out.setdefault(current, [])
            continue

        if current and stripped.startswith("- "):
            value = stripped[2:].strip().strip('"').strip("'")
            if value:
                out[current].append(value)

    return out


def merge_dictionaries(paths: list[Path]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {}
    for path in paths:
        data = load_dictionary(path)
        for key, values in data.items():
            merged.setdefault(key, set()).update(v for v in values if v)
    return {k: sorted(v) for k, v in merged.items()}


def strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(c)
    )


def build_dictionary_replacements(dictionary: dict[str, list[str]]) -> list[tuple[re.Pattern[str], str]]:
    """
    Build conservative replacements.

    This only corrects:
    - accentless forms to accented dictionary entries
    - acronym punctuation variants, e.g. U.N.D.P. -> UNDP
    - exact case variants for all-caps acronyms

    It does not do fuzzy matching. Fuzzy matching creates false positives.
    """
    replacements: list[tuple[re.Pattern[str], str]] = []

    for values in dictionary.values():
        for term in values:
            term = term.strip()
            if not term:
                continue

            accentless = strip_accents(term)
            if accentless != term:
                replacements.append(
                    (re.compile(r"\b" + re.escape(accentless) + r"\b"), term)
                )

            if re.fullmatch(r"[A-Z0-9]{2,}(?:-[A-Z0-9]+)?", term):
                dotted = r"\.".join(re.escape(ch) for ch in term)
                replacements.append(
                    (re.compile(r"\b" + dotted + r"\.?\b", re.IGNORECASE), term)
                )
                replacements.append(
                    (re.compile(r"\b" + re.escape(term.lower()) + r"\b"), term)
                )

    return replacements


DEFAULT_REPLACEMENTS = [
    # Generic UN and captioning fixes. Keep this list narrow.
    (r"\bU\.N\.\b", "UN"),
    (r"\bUnited Nations Development Program\b", "United Nations Development Programme"),
    (r"\bUN Development Program\b", "UN Development Programme"),
    (r"\bSustainable Development Goal\b", "Sustainable Development Goal"),
    (r"\bSustainable Development Goals\b", "Sustainable Development Goals"),
    (r"\bS D Gs\b", "SDGs"),
    (r"\bA I\b", "AI"),
    (r"\bI C T\b", "ICT"),
    (r"\bO S P O\b", "OSPO"),
    (r"\bD P I\b", "DPI"),
    (r"\bD P G\b", "DPG"),
    (r"\bDPI's\b", "DPIs"),
    (r"\bDPG's\b", "DPGs"),
    (r"\be KYC\b", "eKYC"),
    (r"\bE K Y C\b", "eKYC"),
]


def clean_text(
    text: str,
    dictionary_replacements: list[tuple[re.Pattern[str], str]],
    strip_fillers: bool = False,
) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if strip_fillers:
        # Conservative. Do not remove words like "well" or "so".
        text = re.sub(r"\b(?:um|uh|erm)\b[, ]*", "", text, flags=re.IGNORECASE)

    for pattern, replacement in DEFAULT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in dictionary_replacements:
        text = pattern.sub(replacement, text)

    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []

    for part in sentence_parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue

        tokens = re.split(
            r"(\s+(?:and|but|because|so|while|which|that|with|for|from|into|about|through)\s+|,\s+|;\s+|:\s+)",
            part,
        )
        buffer = ""

        for token in tokens:
            if not token:
                continue

            candidate = (buffer + token).strip()
            if not buffer or len(candidate) <= max_chars:
                buffer = candidate
            else:
                chunks.append(buffer.strip(" ,;:"))
                buffer = token.strip()

        if buffer:
            chunks.append(buffer.strip(" ,;:"))

    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(
                textwrap.wrap(
                    chunk,
                    width=max_chars,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

    return [chunk for chunk in final if chunk]


def normalize_cues(
    cues: list[Cue],
    dictionary_replacements: list[tuple[re.Pattern[str], str]],
    max_caption_chars: int,
    min_duration_ms: int,
    merge_gap_ms: int,
    strip_fillers: bool,
) -> list[Cue]:
    cleaned: list[Cue] = []

    for cue in cues:
        if cue.end_ms <= cue.start_ms:
            continue
        text = clean_text(cue.text, dictionary_replacements, strip_fillers)
        if text:
            cleaned.append(Cue(cue.start_ms, cue.end_ms, text))

    # Merge tiny adjacent cues where safe.
    merged: list[Cue] = []
    for cue in cleaned:
        if (
            merged
            and cue.start_ms - merged[-1].end_ms <= merge_gap_ms
            and (merged[-1].end_ms - merged[-1].start_ms < min_duration_ms or len(merged[-1].text) < 25)
            and len(merged[-1].text) + 1 + len(cue.text) <= max_caption_chars
        ):
            merged[-1].end_ms = cue.end_ms
            merged[-1].text = f"{merged[-1].text} {cue.text}"
        else:
            merged.append(cue)

    # Split long cues and distribute timing proportionally.
    split: list[Cue] = []

    for cue in merged:
        parts = split_long_text(cue.text, max_caption_chars)

        if len(parts) == 1:
            split.append(cue)
            continue

        duration = cue.end_ms - cue.start_ms
        total_chars = sum(max(1, len(part)) for part in parts)
        cursor = cue.start_ms

        for idx, part in enumerate(parts):
            if idx == len(parts) - 1:
                end = cue.end_ms
            else:
                end = cursor + max(min_duration_ms, round(duration * len(part) / total_chars))
                remaining = len(parts) - idx - 1
                end = min(end, cue.end_ms - remaining * 500)

            end = max(cursor + 500, end)
            split.append(Cue(cursor, end, part))
            cursor = end

    # Repair overlaps and preserve monotonic order.
    repaired: list[Cue] = []
    last_end = 0

    for cue in split:
        start = max(cue.start_ms, last_end)
        end = max(cue.end_ms, start + 500)
        repaired.append(Cue(start, end, cue.text))
        last_end = end

    return repaired


def wrap_caption(text: str, line_width: int, max_lines: int = 2) -> str:
    lines = textwrap.wrap(
        text,
        width=line_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    if len(lines) <= max_lines:
        return "\n".join(lines)

    # Rebalance into two lines. This avoids three-line captions in most players.
    words = text.split()
    best: tuple[int, str, str] | None = None

    for i in range(1, len(words)):
        first = " ".join(words[:i])
        second = " ".join(words[i:])
        overflow = max(0, len(first) - line_width) + max(0, len(second) - line_width)
        score = abs(len(first) - len(second)) + overflow * 3

        if best is None or score < best[0]:
            best = (score, first, second)

    if best:
        return f"{best[1]}\n{best[2]}"

    return text


def write_vtt(path: Path, cues: list[Cue], source_name: str, line_width: int) -> None:
    lines = [
        "WEBVTT",
        "",
        "NOTE",
        "Generated by normalize_captions.py.",
        "This file has been cleaned for informal caption readability.",
        f"Source: {source_name}",
        "",
    ]

    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{ms_to_ts(cue.start_ms)} --> {ms_to_ts(cue.end_ms)}")
        lines.append(wrap_caption(cue.text, line_width))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_notes(
    path: Path,
    source: Path,
    output: Path,
    input_count: int,
    output_count: int,
    dictionary: dict[str, list[str]],
) -> None:
    lines = [
        "Caption cleanup notes",
        "",
        f"Source file: {source.name}",
        f"Output file: {output.name}",
        f"Input cue count: {input_count}",
        f"Output cue count: {output_count}",
        "",
        "What changed:",
        "- Converted Whisper TXT or rough VTT into valid WebVTT.",
        "- Added cue numbers.",
        "- Merged very short adjacent cues where safe.",
        "- Split long cues into shorter cues.",
        "- Wrapped captions for readability.",
        "- Applied conservative dictionary-based corrections.",
        "",
        "What was not changed:",
        "- No audio re-transcription was performed.",
        "- No event agenda was used.",
        "- No uncertain speaker names were invented.",
        "- No substantive rewriting was performed.",
        "",
        "Dictionary groups loaded:",
    ]

    if not dictionary:
        lines.append("- none")
    else:
        for key, values in sorted(dictionary.items()):
            lines.append(f"- {key}: {len(values)} entries")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize Whisper.cpp TXT/VTT captions into cleaner WebVTT."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input .txt or .vtt files.")
    parser.add_argument(
        "--dictionary",
        action="append",
        type=Path,
        default=[],
        help="YAML or JSON dictionary. Can be supplied multiple times.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("captions"), help="Output directory.")
    parser.add_argument("--max-caption-chars", type=int, default=84)
    parser.add_argument("--line-width", type=int, default=42)
    parser.add_argument("--min-duration-ms", type=int, default=900)
    parser.add_argument("--merge-gap-ms", type=int, default=250)
    parser.add_argument("--strip-fillers", action="store_true")
    parser.add_argument(
        "--suffix",
        default=".accessible.vtt",
        help="Suffix appended to input stem for output VTT.",
    )

    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [path for path in args.dictionary if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing dictionary: {path}", file=sys.stderr)
        return 2

    dictionary = merge_dictionaries(args.dictionary)
    dictionary_replacements = build_dictionary_replacements(dictionary)

    for input_path in args.inputs:
        if not input_path.exists():
            print(f"Missing input: {input_path}", file=sys.stderr)
            continue

        cues = parse_input(input_path)
        if not cues:
            print(f"No cues found: {input_path}", file=sys.stderr)
            continue

        normalized = normalize_cues(
            cues,
            dictionary_replacements=dictionary_replacements,
            max_caption_chars=args.max_caption_chars,
            min_duration_ms=args.min_duration_ms,
            merge_gap_ms=args.merge_gap_ms,
            strip_fillers=args.strip_fillers,
        )

        output_path = args.out_dir / f"{input_path.stem}{args.suffix}"
        notes_path = args.out_dir / f"{input_path.stem}.notes.txt"

        write_vtt(output_path, normalized, input_path.name, args.line_width)
        write_notes(notes_path, input_path, output_path, len(cues), len(normalized), dictionary)

        print(f"Wrote {output_path}")
        print(f"Wrote {notes_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
