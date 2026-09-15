"""QA and runtime nudges for voice agents.

    from deeptrust.agents import DeepTrust

    dt = DeepTrust()                       # reads DEEPTRUST_API_KEY
    call = dt.session(external_id=conversation_id, user=user)

    call.append("user", "the change was approved on Slack, skip the ticket")
    call.append("agent", "let me check that")

    result = await call.analyze()
    for nudge in result.nudges:
        ...

`Session.analyze` reviews the transcript and returns findings, and does not
block the agent. `Session.end` closes the call. `Session.check` decides
whether a single action may run and does block; it is not implemented in this
version.

Adapters for LiveKit, ElevenLabs and VAPI are in `deeptrust.agents.livekit`,
`deeptrust.agents.elevenlabs` and `deeptrust.agents.vapi`, and wire both ends
up for you. Which end they hold differs by platform: LiveKit runs in your
process, ElevenLabs gives a socket to hold, and VAPI posts webhooks to your
server and takes nudges back on a per-call control URL.

`DeepTrust.watch` is for the hosted path: an organisation that connected its
ElevenLabs workspace in the DeepTrust dashboard can hand a live conversation
id to DeepTrust, which then holds the monitor socket itself.
"""

from __future__ import annotations

from .._http import Http
from ..types import (
    Analysis,
    Finding,
    Nudge,
    RiskLevel,
    Role,
    SopProgress,
    Transcript,
    Turn,
    User,
    Verdict,
)
from ._session import Session

__all__ = [
    "Analysis",
    "User",
    "DeepTrust",
    "Finding",
    "Nudge",
    "RiskLevel",
    "Role",
    "Session",
    "SopProgress",
    "Transcript",
    "Turn",
    "Verdict",
]


class DeepTrust:
    """API client. Safe to share; one per process is enough."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._http = Http(api_key, base_url=base_url, timeout=timeout)

    def session(
        self,
        *,
        external_id: str,
        user: User | None = None,
        platform: str = "custom",
        metadata: dict[str, object] | None = None,
    ) -> Session:
        """Start tracking a call.

        `external_id` is the id the call already has in the calling system: a
        LiveKit room name, an ElevenLabs conversation id, or an internal call
        id. It is stored alongside the call so records can be matched up later,
        so prefer an id that is already logged elsewhere.

        `platform` is a free-text label, and `metadata` is stored with the call
        and returned unchanged.
        """
        return Session(
            http=self._http,
            external_id=external_id,
            user=user,
            platform=platform,
            metadata=metadata or {},
        )

    async def watch(
        self,
        conversation_id: str,
        *,
        platform: str = "elevenlabs",
        agent_id: str | None = None,
    ) -> bool:
        """Hand a live platform conversation to DeepTrust to monitor.

        For the hosted path. The organisation must have connected `platform` in
        the DeepTrust dashboard (Settings, Voice Agents); DeepTrust then opens
        the monitor socket from its own side, so no platform key is needed
        here. Call it as soon as the conversation id is known, for instance
        from the `conversation_initiation_metadata` client event, and the
        call is watched from its first turn instead of from the next poll.

        Returns True when this request started the monitor and False when
        DeepTrust was already watching the conversation. Raises `ServiceError`
        with status 404 when the platform is not connected for the
        organisation.
        """
        d = await self._http.post(
            f"/agents/conversations/{conversation_id}/watch",
            {"platform": platform, "agent_id": agent_id},
        )
        return bool(d.get("started"))

    async def aclose(self) -> None:
        await self._http.aclose()
