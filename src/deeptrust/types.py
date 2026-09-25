"""Request and response types for the DeepTrust API.

`Transcript` is a list of `Turn`, and that is also how it goes over the wire.
An agent call has two participants with fixed roles, so a turn always has an
unambiguous speaker. `Transcript.render()` produces the flattened
`"role: text"` form, which some analyses take instead.

`Nudge` has three parts. `title` names what was found, `description` says what
was seen in the call, and `details` says what the agent should do about it. An
agent given only the first two has to choose a response itself, and the choice
it tends to make is to hand the call to a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "agent", "system"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Turn:
    """One thing said on the call.

    `at` is a Unix timestamp and optional. `speaker` names the participant when
    a call has more than the usual two, as after a warm transfer or when a
    supervisor joins; it defaults to `role`.
    """

    role: Role
    text: str
    at: float | None = None
    speaker: str | None = None

    def render(self) -> str:
        who = self.speaker or self.role
        return f"{who}: {self.text}"

    def to_wire(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "text": self.text}
        if self.at is not None:
            d["at"] = self.at
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d


@dataclass
class Transcript:
    """The call so far.

    `append` only adds to this list. Nothing is sent to the API until
    `Session.analyze` is called.
    """

    turns: list[Turn] = field(default_factory=list)

    def append(self, role: Role, text: str, **kw: Any) -> Turn:
        turn = Turn(role=role, text=text, **kw)
        self.turns.append(turn)
        return turn

    def to_wire(self) -> list[dict[str, Any]]:
        """Turns, in order, as sent to the API."""
        return [t.to_wire() for t in self.turns]

    def render(self) -> str:
        """The transcript as one string, a `"role: text"` line per turn."""
        return "\n".join(t.render() for t in self.turns)

    def __len__(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class User:
    """The person the agent is talking to.

    `role` is an authorisation role from the caller's own system, such as
    MEMBER or ADMIN, and policy is written against it. It is unrelated to
    `Turn.role`, which says who spoke.

    Every field is supplied by the integrating application, so `verified` means
    that application considers this person verified. DeepTrust records the
    claim and does not check it.
    """

    id: str
    role: str = "MEMBER"
    name: str | None = None
    verified: bool = False
    verified_via: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "name": self.name,
            "verified": self.verified,
            "verified_via": self.verified_via,
        }


@dataclass(frozen=True)
class Nudge:
    """Something to tell the agent mid-call, with the action to take.

    `id` identifies the nudge across the ways it can reach an agent: the same
    nudge carries the same id on an analyze response and when DeepTrust pushes
    it into a call. Older API versions do not send one, so it may be None.
    """

    title: str
    description: str
    details: str
    id: str | None = None

    def render(self) -> str:
        """`description` and `details` joined, for platforms that accept a
        single block of context rather than fields."""
        return " ".join(p for p in (self.description, self.details) if p).strip()


@dataclass(frozen=True)
class Finding:
    """Something the analysis found in the call.

    `nudge` is set only when the finding is worth telling the agent about.
    """

    kind: str
    detail: str
    sop_id: str | None = None
    control: str | None = None
    risk_level: RiskLevel | None = None
    confidence: float | None = None
    nudge: Nudge | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SopProgress:
    """Where the call has got to in a procedure."""

    sop_id: str
    name: str
    applicable: bool
    in_progress: bool
    being_followed: bool
    steps_completed: list[int] = field(default_factory=list)
    steps_total: int = 0


@dataclass(frozen=True)
class Analysis:
    """The result of one job over the transcript."""

    session_id: str
    job_id: str
    findings: list[Finding] = field(default_factory=list)
    progress: list[SopProgress] = field(default_factory=list)
    risk_level: RiskLevel | None = None
    confidence: float | None = None
    reasoning: str | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def nudges(self) -> list[Nudge]:
        """The nudges from this analysis, in the order the findings came."""
        return [f.nudge for f in self.findings if f.nudge is not None]


@dataclass(frozen=True)
class Verdict:
    """A decision about one action the agent wants to take.

    `blocked` is true when the action must not run. `resolution` says how the
    call should proceed instead, and `instruction` is the refusal and the
    resolution written as a single sentence to pass to a model.

    Returned by `Session.check`, which is not yet implemented.
    """

    decision: Literal["allow", "warn", "deny", "hold"]
    blocked: bool
    reason: str
    resolution: Literal["resolve_on_call", "ticket", "handover", "none"]
    instruction: str
    control: str | None = None
    message: str | None = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
