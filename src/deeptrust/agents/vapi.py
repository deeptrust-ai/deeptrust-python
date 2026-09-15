"""VAPI adapter.

    from deeptrust.agents import DeepTrust
    from deeptrust.agents.vapi import Bridge

    bridge = Bridge(DeepTrust(), api_key=os.environ["VAPI_API_KEY"])

    @app.post("/vapi/webhook")             # your route, your server
    async def vapi_webhook(payload: dict):
        await bridge.handle(payload, user=caller)
        return {}

VAPI's transport is the mirror image of ElevenLabs'. There is no socket anyone
can hold open: VAPI posts its server-url events to *your* server, and what goes
back the other way goes to a per-call HTTPS endpoint VAPI mints for that call
and publishes on the call object as `monitor.controlUrl`. So the adapter is a
handler you call from inside your own webhook route rather than a watcher with
a loop of its own, and it needs no code inside your agent either way.

A nudge is delivered as `add-message` with `triggerResponseEnabled: true`,
which is an interrupt: VAPI hands the system message to the model and has it
respond immediately, cutting into what the agent is saying. That makes VAPI
behave like the LiveKit adapter rather than the ElevenLabs one, whose
contextual update is documented as non-interrupting and only shapes the turn
after the current one. A system message rather than a `say` because a `say`
would put our words in the agent's mouth verbatim, while a system message lets
the agent's own persona carry them.

The control URL is read from the webhook payload when the event carries it,
and fetched with `GET /call/{id}` when it does not, then cached for the rest of
the call. Inbound calls are the case this exists for: nobody placed the call,
so there is no creation-time response to have captured a URL from, and an
adapter that assumed one would work for outbound calls only.

`monitor.listenUrl` sits next to it and is deliberately ignored: it is a raw
PCM audio stream, not a channel anything can be sent on.

A control URL is only accepted if it is HTTPS on VAPI's own domain. The webhook
body is attacker-reachable in the general case -- it arrives over the public
internet at your route -- and a nudge names what DeepTrust found in the call, so
a forged `monitor.controlUrl` would be a way to have this SDK post that text to
a host of someone else's choosing. Anything off `vapi.ai` reads as no control
URL rather than as an error.

Only final transcripts are read. VAPI emits a `transcript` event per partial as
the sentence is still being recognised, and analysing those re-analyses the
same sentence several times -- the same class of bug the LiveKit adapter's
`last` dict guards against, arriving here by a different route.

VAPI's `tool-calls` webhook is the one event whose response controls what the
agent does next, and this adapter does not answer it. Blocking a tool call is
`Session.check`, which is separate work; this is transcript in, nudge out.

No extra dependency: httpx is already the client's own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from ..errors import ConfigError
from ..types import Analysis, Nudge, User
from . import DeepTrust
from ._session import Session

API_BASE_URL = "https://api.vapi.ai"

#: The only domain a control URL may point at.
CONTROL_URL_DOMAIN = "vapi.ai"


def add_message_command(text: str) -> dict[str, Any]:
    """The control-URL body that delivers a nudge as an interrupt.

    `triggerResponseEnabled` is what makes it one. Without it VAPI appends the
    message and waits for the agent to reach its next turn on its own, which
    is a different product: the caller is being worked on now.
    """
    return {
        "type": "add-message",
        "message": {"role": "system", "content": text},
        "triggerResponseEnabled": True,
    }


class Bridge:
    """Turns VAPI server-url events into DeepTrust calls, and nudges into
    messages on the live call.

    One bridge serves every call your server receives; state is kept per VAPI
    call id, and dropped when the call reports it ended.
    """

    def __init__(
        self,
        dt: DeepTrust,
        *,
        api_key: str,
        deliver: bool = True,
        on_analysis: Callable[[Analysis], None] | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """`api_key` is a VAPI private key: it reads the call object to find
        the control URL when an event does not carry one."""
        if not api_key:
            raise ConfigError(
                "Bridge needs a VAPI private API key. It reads the call to "
                "find monitor.controlUrl, which is where a nudge is sent."
            )
        self._dt = dt
        self._key = api_key
        self._deliver = deliver
        self._on_analysis = on_analysis
        self._base_url = base_url
        self._sessions: dict[str, Session] = {}
        # Per call, because VAPI mints the URL per call. Cached because most
        # events carry it and the fetch is only for the ones that do not.
        self._control: dict[str, str] = {}

    async def handle(
        self,
        payload: dict[str, Any],
        *,
        user: User | None = None,
    ) -> Analysis | None:
        """Process one server-url event. Returns the analysis it caused, if any.

        Call it for every event and let it decide: events that are not turns
        cost nothing, and the ones that are not transcripts still carry the
        call object the control URL is learned from.

        Returns None for an event that started no job, which is most of them.
        """
        message = _message(payload)
        call = message.get("call")
        call = call if isinstance(call, dict) else {}
        call_id = str(call.get("id") or "")
        if not call_id:
            return None

        url = _monitor_control_url(call)
        if url:
            self._control[call_id] = url

        kind = message.get("type")
        if kind == "end-of-call-report":
            await self._finish(call_id)
            return None
        if kind != "transcript":
            return None

        role, text = _read_turn(message)
        if not text:
            return None

        session = self._session(call_id, user)
        session.append(role, text)

        # Caller turns only. Feeding the agent's own replies back in doubles
        # the work and lets its answers reclassify the call.
        if role != "user":
            return None

        result = await session.analyze()
        if result is None:
            return None
        if self._on_analysis:
            self._on_analysis(result)
        if self._deliver:
            for nudge in result.nudges:
                await self.send_nudge(call_id, nudge)
        return result

    async def send_nudge(self, call_id: str, nudge: Nudge) -> bool:
        """Send one nudge into a live call. Returns whether VAPI took it.

        A call that has already ended publishes no control URL, so a nudge
        produced from its last turn returns False rather than raising: the
        call it was for is over, and the finding is already recorded.
        """
        url = await self.control_url(call_id)
        if not url:
            return False

        async with httpx.AsyncClient() as client:
            # No credential on this request. The control URL carries its own
            # authority and VAPI does not accept the private key here.
            response = await client.post(url, json=add_message_command(nudge.render()))
        return response.is_success

    async def control_url(self, call_id: str) -> str | None:
        """The call's `monitor.controlUrl`, from cache or from VAPI."""
        cached = self._control.get(call_id)
        if cached:
            return cached

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._key}"},
        ) as client:
            # Quoted: the id comes off a webhook body, and a raw `/` or `?` in
            # it would address a different endpoint of the API than the call
            # lookup.
            response = await client.get(f"/call/{quote(call_id, safe='')}")
        if not response.is_success:
            return None

        body = response.json()
        url = _monitor_control_url(body if isinstance(body, dict) else {})
        if url:
            self._control[call_id] = url
        return url

    def session(self, call_id: str) -> Session | None:
        """The DeepTrust session for a call, so its transcript stays reachable."""
        return self._sessions.get(call_id)

    def _session(self, call_id: str, user: User | None) -> Session:
        session = self._sessions.get(call_id)
        if session is None:
            session = self._dt.session(external_id=call_id, user=user, platform="vapi")
            self._sessions[call_id] = session
        return session

    async def _finish(self, call_id: str) -> None:
        self._control.pop(call_id, None)
        session = self._sessions.pop(call_id, None)
        if session is not None:
            await session.end()


def _message(payload: dict[str, Any]) -> dict[str, Any]:
    """The event itself, out of the request body.

    VAPI wraps a server-url event in `{"message": {...}}`. A bare event is
    accepted too, so a payload already unwrapped by the caller's own framework
    still works.
    """
    message = payload.get("message")
    return message if isinstance(message, dict) else payload


def _read_turn(message: dict[str, Any]) -> tuple[str, str]:
    """A turn from a transcript event, or ("", "") if it is not one yet.

    Partials are not turns. VAPI sends one event per revision of the sentence
    being recognised, all with the same `transcriptType: "partial"`, and only
    the final one is the sentence the caller actually said.
    """
    if message.get("transcriptType") != "final":
        return "", ""
    role = "user" if str(message.get("role") or "") == "user" else "agent"
    return role, str(message.get("transcript") or "").strip()


def _monitor_control_url(call: dict[str, Any]) -> str | None:
    """`monitor.controlUrl` off a call object, or None.

    Defensive about the shape rather than trusting it: this runs on the
    webhook path, and a missing or renamed field has to read as "no control
    URL yet" -- which a fetch may still answer -- instead of as a TypeError
    inside the customer's webhook route.
    """
    monitor = call.get("monitor")
    if not isinstance(monitor, dict):
        return None
    url = monitor.get("controlUrl")
    if not isinstance(url, str) or not url:
        return None
    return url if _is_vapi_control_url(url) else None


def _is_vapi_control_url(url: str) -> bool:
    """Whether a URL is one VAPI could have minted: HTTPS, on `vapi.ai`.

    The check is on the host rather than the full URL because VAPI mints these
    per region and per call -- the path and the subdomain both vary -- while
    the domain is the part that says the destination is VAPI and not somewhere
    a forged webhook pointed us.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host == CONTROL_URL_DOMAIN or host.endswith(f".{CONTROL_URL_DOMAIN}")
