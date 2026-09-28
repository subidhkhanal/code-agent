"""Per-task logs of exactly what was sent to the model, for debugging and retrieval evals.

Layout: `.agent/logs/<request_id>/`
  retrieval.json   queries, chosen chunks with scores and sources, what was dropped/collapsed
  context.md       the rendered context block, as the model saw it
  requests.jsonl   every outbound LLM request (model, messages, tool names), one per line

The request log is installed as the gateway's outbound filter, the last step before a request
leaves the machine, so it records what was actually sent. Secret redaction (M3) runs earlier in
the same chain, so these logs only ever contain redacted text. `.agent/` is a protected path:
the agent's own tools cannot read or edit these logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from code_agent.context.assembly import ContextBundle
from code_agent.llm.types import Request


class TaskLog:
    def __init__(self, state_dir: Path) -> None:
        self.root = state_dir / "logs"
        self.request_id: str | None = None

    @property
    def dir(self) -> Path | None:
        return None if self.request_id is None else self.root / self.request_id

    def start(self, request_id: str) -> None:
        self.request_id = request_id
        assert self.dir is not None
        self.dir.mkdir(parents=True, exist_ok=True)

    def retrieval(self, bundle: ContextBundle) -> None:
        if self.dir is None:
            return
        (self.dir / "context.md").write_text(bundle.render(), encoding="utf-8")
        record = {
            "queries": bundle.queries,
            "budget_tokens": bundle.budget_tokens,
            "tokens": bundle.tokens,
            "collapsed": bundle.collapsed,
            "items": [
                {
                    "file_path": i.chunk.file_path,
                    "lines": [i.chunk.start_line, i.chunk.end_line],
                    "symbol": i.chunk.symbol,
                    "kind": i.chunk.kind,
                    "score": round(i.score, 6),
                    "source": i.source,
                    "signature_only": i.signature_only,
                }
                for i in bundle.items
            ],
            "dropped": [
                {"file_path": c.file_path, "lines": [c.start_line, c.end_line], "symbol": c.symbol}
                for c in bundle.dropped
            ],
            "base_hashes": bundle.base_hashes,
        }
        (self.dir / "retrieval.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def __call__(self, request: Request) -> Request:
        """Gateway outbound filter: record the request, return it unchanged."""
        if self.dir is not None:
            line = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "model": request.model,
                "max_output_tokens": request.max_output_tokens,
                "tools": [t.name for t in request.tools],
                "messages": [
                    {k: v for k, v in asdict(m).items() if v not in (None, "", ())}
                    for m in request.messages
                ],
            }
            with (self.dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
        return request
