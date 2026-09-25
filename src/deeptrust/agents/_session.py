"""A single call, and the two requests that can be made about it.

The session accumulates turns locally and submits the transcript when
`analyze` is called. Results are not stored on the session: each call to
`analyze` returns its own `Analysis`, and the full record lives server-side.
`end` closes the call so post-call processing starts now rather than after
the server's inactivity timeout.
"""

from __future__ import annotations

import time
from typing import Any

from .._http import Http
from ..types import (
    Analysis,
    Finding,
    Nudge,
    SopProgress,
    Transcript,
    Turn,
    User,
    Verdict,
)


def _nudge(d: dict[str, Any] | None) -> Nudge | None:
    if not d:
        return None
    return Nudge(
        title=str(d.get("title") or ""),
        description=str(d.get("description") or ""),
        details=str(d.get("details") or ""),
        id=str(d["id"]) if d.get("id") else None,
    )


def _finding(d: dict[str, Any]) -> Finding:
    return Finding(
        kind=str(d.get("kind") or "finding"),
        detail=str(d.get("detail") or ""),
        sop_id=d.get("sop_id"),
        control=d.get("control"),
        risk_level=d.get("risk_level"),
        confidence=d.get("confidence"),
        nudge=_nudge(d.get("nudge")),
        raw=d,
    )


def _progress(d: dict[str, Any]) -> SopProgress:
    return SopProgress(
        sop_id=str(d.get("sop_id") or ""),
        name=str(d.get("name") or ""),
        applicable=bool(d.get("applicable")),
        in_progress=bool(d.get("in_progress")),
        being_followed=bool(d.get("being_followed", True)),
        steps_completed=list(d.get("steps_completed") or []),
        steps_total=int(d.get("steps_total") or 0),
    )


class Session:
    def __init__(
        self,
        *,
        http: Http,
        external_id: str,
        user: User | None,
        platform: str,
        metadata: dict[str, object],
    ) -> None:
        self._http = http
        self.external_id = external_id
        self.user = user
        self.platform = platform
        self.metadata = metadata
        self.transcript = Transcript()
        # Assigned by the API on the first analyze(), then sent with every
        # later request so they group into one call.
        self.id: str | None = None
        # Turn count at the last analyze(), used to skip a request when
        # nothing new has been said.
        self._analyzed_upto = 0

    # ── building the transcript ──────────────────────────────────────────────

    def append(self, role: str, text: str, **kw: Any) -> Turn:
        """Add a turn to the transcript. Sends nothing."""
        return self.transcript.append(role, text, **kw)  # type: ignore[arg-type]

    @property
    def pending(self) -> int:
        """Turns added since the last `analyze`."""
        return len(self.transcript) - self._analyzed_upto

    # ── the semantic plane ───────────────────────────────────────────────────

    async def analyze(self, *, force: bool = False) -> Analysis | None:
        """Analyze the transcript and return what was found.

        Returns None when no turns have been added since the last call, so
        this can be called on every turn without sending a request each time.
        Pass `force=True` to analyze regardless.

        This does not block the agent. A result arrives after the turn that
        produced it has already been spoken, so a nudge affects what the agent
        says next rather than what it is saying now.
        """
        if not force and self.pending == 0:
            return None

        # The whole transcript is sent each time. The API tracks what it has
        # already seen for this session and analyzes only the new turns.
        t0 = time.perf_counter()
        body: dict[str, Any] = {
            "external_id": self.external_id,
            "platform": self.platform,
            "turns": self.transcript.to_wire(),
            "metadata": self.metadata,
        }
        if self.id:
            body["session_id"] = self.id
        if self.user:
            body["user"] = self.user.to_wire()

        d = await self._http.post("/agents/analyze", body)
        self.id = d.get("session_id") or self.id
        self._analyzed_upto = len(self.transcript)

        return Analysis(
            session_id=str(self.id or ""),
            job_id=str(d.get("job_id") or ""),
            findings=[_finding(f) for f in d.get("findings") or []],
            progress=[_progress(p) for p in d.get("progress") or []],
            risk_level=d.get("risk_level"),
            confidence=d.get("confidence"),
            reasoning=d.get("reasoning"),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            raw=d,
        )

    async def end(self) -> bool:
        """Tell DeepTrust the call is over.

        Post-call processing starts at once instead of after the server's
        inactivity timeout, so the record is complete minutes sooner. Returns
        True when this request ended the call and False when it was already
        ended, or when nothing was ever analyzed (there is no call to end).
        Calling it twice is harmless.
        """
        if not self.id:
            return False
        d = await self._http.post(f"/agents/sessions/{self.id}/end")
        return bool(d.get("ended")) and not bool(d.get("already_ended"))

    # ── the action plane ─────────────────────────────────────────────────────

    async def check(
        self,
        *,
        action: str,
        args: dict[str, Any] | None = None,
        facts: dict[str, Any] | None = None,
    ) -> Verdict:
        """Decide whether an action may run. Not yet implemented.

        Unlike `analyze`, this blocks: it is meant to be called from a tool
        handler before the action executes, and the returned `Verdict` says
        whether to proceed.

        `facts` carries the values the policy is written against, such as
        whether a change ticket is approved or an account is protected.
        Policies compare these fields rather than reading the transcript, which
        is what makes the decision deterministic, so the calling application
        computes them from its own systems before proposing the action.
        """
        raise NotImplementedError(
            "Session.check is not implemented in this version. "
            "This release covers analysis and nudge delivery."
        )
