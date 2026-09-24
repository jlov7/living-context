"""Generate the authored revision corpus (Milestone A data).

24 documents, 6 families, 3 chronological revisions each (72 revision
entries). The families exercise every edit kind the protocol requires:
changed numbers, renamed entities, revoked facts, contradictory updates,
cross-document references, and unchanged controls. All content is synthetic
technical fiction — NO real employer/client/personal data by design.

Run: uv run python experiments/generate_revision_sequences.py
Output: experiments/revision_sequences.jsonl (deterministic, stable order)
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "revision_sequences.jsonl"

FAMILIES = ["hydraulics", "telemetry", "compliance", "logistics", "supply", "controls"]


def _mk(
    family: str,
    doc: str,
    rev: int,
    kind: str,
    as_of: str,
    content: str,
) -> dict:
    return {
        "doc_id": f"{family}/{doc}",
        "family": family,
        "revision": f"v{rev}",
        "edit_kind": kind,
        "as_of": as_of,
        "content": content,
    }


def _seq() -> list[dict]:
    rows: list[dict] = []

    # --- family: hydraulics — changed numbers ---------------------------------
    # h-pump: flow spec changes twice. h-valve: pressure rating changes once.
    # h-seal: unchanged control. h-tank: capacity changes once.
    rows += [
        _mk(
            "hydraulics",
            "h-pump",
            1,
            "initial",
            "2026-01-05",
            "H-1 pump. Flow 40 L/min at 150 bar. Inlet size 25 mm.",
        ),
        _mk(
            "hydraulics",
            "h-pump",
            2,
            "number-change",
            "2026-04-12",
            "H-1 pump. Flow 55 L/min at 150 bar. Inlet size 25 mm.",
        ),
        _mk(
            "hydraulics",
            "h-pump",
            3,
            "number-change",
            "2026-07-28",
            "H-1 pump. Flow 55 L/min at 165 bar. Inlet size 25 mm.",
        ),
        _mk(
            "hydraulics",
            "h-valve",
            1,
            "initial",
            "2026-01-05",
            "V-2 valve. Rated 160 bar, pilot-operated.",
        ),
        _mk(
            "hydraulics",
            "h-valve",
            2,
            "number-change",
            "2026-04-12",
            "V-2 valve. Rated 200 bar, pilot-operated.",
        ),
        _mk(
            "hydraulics",
            "h-valve",
            3,
            "unchanged",
            "2026-07-28",
            "V-2 valve. Rated 200 bar, pilot-operated.",
        ),
        _mk(
            "hydraulics",
            "h-seal",
            1,
            "initial",
            "2026-01-05",
            "S-3 seal. PTFE base, temperature limit 120 C.",
        ),
        _mk(
            "hydraulics",
            "h-seal",
            2,
            "unchanged",
            "2026-04-12",
            "S-3 seal. PTFE base, temperature limit 120 C.",
        ),
        _mk(
            "hydraulics",
            "h-seal",
            3,
            "unchanged",
            "2026-07-28",
            "S-3 seal. PTFE base, temperature limit 120 C.",
        ),
        _mk(
            "hydraulics",
            "h-tank",
            1,
            "initial",
            "2026-01-05",
            "T-4 tank. Volume 60 L, ambient fill only.",
        ),
        _mk(
            "hydraulics",
            "h-tank",
            2,
            "number-change",
            "2026-04-12",
            "T-4 tank. Volume 60 L, ambient fill only. Drain at bottom.",
        ),
        _mk(
            "hydraulics",
            "h-tank",
            3,
            "number-change",
            "2026-07-28",
            "T-4 tank. Volume 45 L, ambient fill only. Drain at bottom.",
        ),
    ]

    # --- family: telemetry — renamed entities ----------------------------------
    # t-gw: gateway renamed twice. t-rate: sampling renamed once.
    # t-clock: unchanged control. t-chan: channel renamed once.
    rows += [
        _mk(
            "telemetry",
            "t-gw",
            1,
            "initial",
            "2026-01-05",
            "Gateway GW-1 aggregates site telemetry every 60 s.",
        ),
        _mk(
            "telemetry",
            "t-gw",
            2,
            "rename",
            "2026-04-12",
            "Gateway GATE-NORTH aggregates site telemetry every 60 s.",
        ),
        _mk(
            "telemetry",
            "t-gw",
            3,
            "rename",
            "2026-07-28",
            "Gateway GATE-CENTRAL aggregates site telemetry every 60 s.",
        ),
        _mk(
            "telemetry",
            "t-rate",
            1,
            "initial",
            "2026-01-05",
            "Sampling rate 10 Hz; default retention 90 days.",
        ),
        _mk(
            "telemetry",
            "t-rate",
            2,
            "rename",
            "2026-04-12",
            "Sampling rate 10 Hz; default retention 90 days. Field renamed ping_rate.",
        ),
        _mk(
            "telemetry",
            "t-rate",
            3,
            "unchanged",
            "2026-07-28",
            "Sampling rate 10 Hz; default retention 90 days. Field renamed ping_rate.",
        ),
        _mk(
            "telemetry",
            "t-clock",
            1,
            "initial",
            "2026-01-05",
            "Timestamps in UTC, NTP synchronized, drift tolerance 50 ms.",
        ),
        _mk(
            "telemetry",
            "t-clock",
            2,
            "unchanged",
            "2026-04-12",
            "Timestamps in UTC, NTP synchronized, drift tolerance 50 ms.",
        ),
        _mk(
            "telemetry",
            "t-clock",
            3,
            "unchanged",
            "2026-07-28",
            "Timestamps in UTC, NTP synchronized, drift tolerance 50 ms.",
        ),
        _mk(
            "telemetry",
            "t-chan",
            1,
            "initial",
            "2026-01-05",
            "Channel CH-A reserved for vibration data.",
        ),
        _mk(
            "telemetry",
            "t-chan",
            2,
            "rename",
            "2026-04-12",
            "Channel CH-ACOUSTIC reserved for vibration data.",
        ),
        _mk(
            "telemetry",
            "t-chan",
            3,
            "rename",
            "2026-07-28",
            "Channel CH-SONIC reserved for vibration data.",
        ),
    ]

    # --- family: compliance — revoked/corrected facts --------------------------
    # c-batch: batch release revoked once, re-revoked later.
    # c-label: labeling rule corrected. c-doc: unchanged control. c-audit: audit window changed once.
    rows += [
        _mk(
            "compliance",
            "c-batch",
            1,
            "initial",
            "2026-01-05",
            "Batch B-120 released 2026-01-03. Release status: APPROVED.",
        ),
        _mk(
            "compliance",
            "c-batch",
            2,
            "revoked-fact",
            "2026-04-12",
            "Batch B-120 previously approved is REVOKED pending re-test.",
        ),
        _mk(
            "compliance",
            "c-batch",
            3,
            "revoked-fact",
            "2026-07-28",
            "Batch B-120 release remains REVOKED; re-test failed twice.",
        ),
        _mk(
            "compliance",
            "c-label",
            1,
            "initial",
            "2026-01-05",
            "Label must state max fill pressure 150 bar.",
        ),
        _mk(
            "compliance",
            "c-label",
            2,
            "correction",
            "2026-04-12",
            "Label must state max fill pressure 200 bar (correction).",
        ),
        _mk(
            "compliance",
            "c-label",
            3,
            "unchanged",
            "2026-07-28",
            "Label must state max fill pressure 200 bar (correction).",
        ),
        _mk(
            "compliance",
            "c-doc",
            1,
            "initial",
            "2026-01-05",
            "Document control: one week review after change.",
        ),
        _mk(
            "compliance",
            "c-doc",
            2,
            "unchanged",
            "2026-04-12",
            "Document control: one week review after change.",
        ),
        _mk(
            "compliance",
            "c-doc",
            3,
            "unchanged",
            "2026-07-28",
            "Document control: one week review after change.",
        ),
        _mk(
            "compliance",
            "c-audit",
            1,
            "initial",
            "2026-01-05",
            "Audit window quarterly; sampling 5% of lots.",
        ),
        _mk(
            "compliance",
            "c-audit",
            2,
            "number-change",
            "2026-04-12",
            "Audit window quarterly; sampling 10% of lots.",
        ),
        _mk(
            "compliance",
            "c-audit",
            3,
            "unchanged",
            "2026-07-28",
            "Audit window quarterly; sampling 10% of lots.",
        ),
    ]

    # --- family: logistics — deleted IDs + overlapping updates -----------------
    # l-route: route renamed then deleted. l-depot: depot closed (deleted).
    # l-mode: unchanged control. l-rate: rate changed overlapping close of depot.
    rows += [
        _mk(
            "logistics", "l-route", 1, "initial", "2026-01-05", "Route R-7 serves depots D1 and D2."
        ),
        _mk(
            "logistics", "l-route", 2, "rename", "2026-04-12", "Route R-7B serves depots D1 and D2."
        ),
        _mk(
            "logistics",
            "l-route",
            3,
            "deleted",
            "2026-07-28",
            "Route R-7B retired; depots D1 and D2 served by R-9.",
        ),
        _mk(
            "logistics",
            "l-depot",
            1,
            "initial",
            "2026-01-05",
            "Depot D2 open; receives daily loads.",
        ),
        _mk(
            "logistics",
            "l-depot",
            2,
            "unchanged",
            "2026-04-12",
            "Depot D2 open; receives daily loads.",
        ),
        _mk(
            "logistics",
            "l-depot",
            3,
            "deleted",
            "2026-07-28",
            "Depot D2 CLOSED; loads rerouted to D3.",
        ),
        _mk(
            "logistics", "l-mode", 1, "initial", "2026-01-05", "Standard mode is rail for > 500 km."
        ),
        _mk(
            "logistics",
            "l-mode",
            2,
            "unchanged",
            "2026-04-12",
            "Standard mode is rail for > 500 km.",
        ),
        _mk(
            "logistics",
            "l-mode",
            3,
            "unchanged",
            "2026-07-28",
            "Standard mode is rail for > 500 km.",
        ),
        _mk("logistics", "l-rate", 1, "initial", "2026-01-05", "Rate per pallet 12 USD via D2."),
        _mk(
            "logistics",
            "l-rate",
            2,
            "number-change",
            "2026-04-12",
            "Rate per pallet 14 USD via D3.",
        ),
        _mk(
            "logistics",
            "l-rate",
            3,
            "number-change",
            "2026-07-28",
            "Rate per pallet 11 USD via D3.",
        ),
    ]

    # --- family: supply — contradictory updates + cross-document refs ----------
    # s-spec: spec contradicts s-price. s-quote: quote references spec revision.
    # s-status: unchanged control. s-capacity: capacity changes contradictory to s-quote.
    rows += [
        _mk("supply", "s-spec", 1, "initial", "2026-01-05", "Spec SP-1: alloy AL-2024, temper T6."),
        _mk(
            "supply",
            "s-spec",
            2,
            "contradictory-update",
            "2026-04-12",
            "Spec SP-1: alloy AL-7075, temper T6. (Supersedes AL-2024.)",
        ),
        _mk(
            "supply",
            "s-spec",
            3,
            "contradictory-update",
            "2026-07-28",
            "Spec SP-1: alloy AL-7075, temper T651. (Supersedes T6.)",
        ),
        _mk(
            "supply",
            "s-quote",
            1,
            "initial",
            "2026-01-05",
            "Quote Q-1 valid 90 days; references SP-1 revision v1.",
        ),
        _mk(
            "supply",
            "s-quote",
            2,
            "cross-reference",
            "2026-04-12",
            "Quote Q-1 revised; references SP-1 revision v2; price 4.2 USD/kg.",
        ),
        _mk(
            "supply",
            "s-quote",
            3,
            "contradictory-update",
            "2026-07-28",
            "Quote Q-1 revised; references SP-1 revision v3; price 5.0 USD/kg.",
        ),
        _mk(
            "supply",
            "s-status",
            1,
            "initial",
            "2026-01-05",
            "Supplier status: qualified for structural work.",
        ),
        _mk(
            "supply",
            "s-status",
            2,
            "unchanged",
            "2026-04-12",
            "Supplier status: qualified for structural work.",
        ),
        _mk(
            "supply",
            "s-status",
            3,
            "unchanged",
            "2026-07-28",
            "Supplier status: qualified for structural work.",
        ),
        _mk(
            "supply",
            "s-capacity",
            1,
            "initial",
            "2026-01-05",
            "Foundry capacity 40 t/month; supports quote Q-1.",
        ),
        _mk(
            "supply",
            "s-capacity",
            2,
            "number-change",
            "2026-04-12",
            "Foundry capacity 55 t/month; supports quote Q-1.",
        ),
        _mk(
            "supply",
            "s-capacity",
            3,
            "number-change",
            "2026-07-28",
            "Foundry capacity 55 t/month; price basis 5.0 USD/kg.",
        ),
    ]

    # --- family: controls — intended as unchanged controls -----------------------
    # k-id: unchanged. k-seq: unchanged. k-scope: unchanged. k-sign: unchanged.
    rows += [
        _mk("controls", "k-id", 1, "initial", "2026-01-05", "System ID format: SS-KKKK-NNN."),
        _mk("controls", "k-id", 2, "unchanged", "2026-04-12", "System ID format: SS-KKKK-NNN."),
        _mk("controls", "k-id", 3, "unchanged", "2026-07-28", "System ID format: SS-KKKK-NNN."),
        _mk(
            "controls",
            "k-seq",
            1,
            "initial",
            "2026-01-05",
            "Sequence numbering starts at 1000 per site.",
        ),
        _mk(
            "controls",
            "k-seq",
            2,
            "unchanged",
            "2026-04-12",
            "Sequence numbering starts at 1000 per site.",
        ),
        _mk(
            "controls",
            "k-seq",
            3,
            "unchanged",
            "2026-07-28",
            "Sequence numbering starts at 1000 per site.",
        ),
        _mk(
            "controls",
            "k-scope",
            1,
            "initial",
            "2026-01-05",
            "Scope excludes maintenance accessories.",
        ),
        _mk(
            "controls",
            "k-scope",
            2,
            "unchanged",
            "2026-04-12",
            "Scope excludes maintenance accessories.",
        ),
        _mk(
            "controls",
            "k-scope",
            3,
            "unchanged",
            "2026-07-28",
            "Scope excludes maintenance accessories.",
        ),
        _mk("controls", "k-sign", 1, "initial", "2026-01-05", "Sign-off requires two operators."),
        _mk("controls", "k-sign", 2, "unchanged", "2026-04-12", "Sign-off requires two operators."),
        _mk("controls", "k-sign", 3, "unchanged", "2026-07-28", "Sign-off requires two operators."),
    ]

    assert len(rows) == 24 * 3, len(rows)
    return rows


def main() -> None:
    rows = _seq()
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} revision entries -> {OUT}")


if __name__ == "__main__":
    main()
