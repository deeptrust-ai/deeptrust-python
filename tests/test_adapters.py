"""Adapter tests, with the platform SDKs faked.

The point of these is the wiring, not the platform: analysis runs on caller
turns only, and a nudge is delivered exactly once per finding that carries one.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx

from deeptrust.agents import DeepTrust
from deeptrust.agents.elevenlabs import Monitor, _read_turn, contextual_update_command
from deeptrust.agents.livekit import attach
from deeptrust.agents.vapi import Bridge, add_message_command
from deeptrust.agents.vapi import _read_turn as _vapi_read_turn

BASE = "https://example.test/api/v1"
# Shaped like the real thing: VAPI mints these per region and per call, and only
# the domain is fixed.
CONTROL_URL = (
    "https://aws-us-west-2-production1-phone-call-websocket.vapi.ai/call_1/control"
)

ONE_NUDGE = {
    "session_id": "sess_1",
    "job_id": "job_1",
    "findings": [
        {
            "kind": "coercion",
            "detail": "third party instructing",
            "nudge": {
                "title": "Someone else may be coaching the caller",
                "description": "The caller referred to someone else on the line.",
                "details": "Ask one question and wait: is anyone helping them right now?",
            },
        }
    ],
}


class FakeAgent:
    def __init__(self) -> None:
        self.chat_ctx = FakeChat()
        self.updated: list[Any] = []

    async def update_chat_ctx(self, chat: Any) -> None:
        self.updated.append(chat)


class FakeChat:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def copy(self) -> FakeChat:
        c = FakeChat()
        c.messages = list(self.messages)
        return c

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


class FakeSession:
    """Stands in for a LiveKit AgentSession."""

    def __init__(self) -> None:
        self.current_agent = FakeAgent()
        self.handlers: dict[str, Any] = {}
        self.interrupted = 0
        self.replies: list[str] = []

    def on(self, event: str) -> Any:
        def deco(fn: Any) -> Any:
            self.handlers[event] = fn
            return fn

        return deco

    def interrupt(self) -> None:
        self.interrupted += 1

    def generate_reply(self, instructions: str) -> None:
        self.replies.append(instructions)

    async def say(self, role: str, text: str) -> None:
        item = type("Item", (), {"text_content": text, "role": role})()
        ev = type("Ev", (), {"item": item})()
        self.handlers["conversation_item_added"](ev)


@respx.mock
async def test_livekit_analyses_caller_turns_and_delivers_once() -> None:
    route = respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    lk = FakeSession()
    dt = DeepTrust(api_key="dt_test", base_url=BASE)
    call = attach(lk, dt, external_id="room-1")

    await lk.say("user", "my colleague is telling me what to say")
    # The handler spawns a task; let it run.
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0.05)

    assert route.call_count == 1
    assert len(call.transcript) == 1
    assert lk.interrupted == 1
    assert len(lk.replies) == 1
    assert "Ask one question" in lk.replies[0]
    # The nudge went into the agent's context as well as being spoken.
    assert lk.current_agent.updated


@respx.mock
async def test_livekit_does_not_analyse_the_agents_own_turn() -> None:
    route = respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    lk = FakeSession()
    dt = DeepTrust(api_key="dt_test", base_url=BASE)
    call = attach(lk, dt, external_id="room-1")

    await lk.say("assistant", "I need to confirm it is you first")
    import asyncio

    await asyncio.sleep(0.05)

    assert route.call_count == 0
    # It is still in the transcript. It is just not a reason to run a job.
    assert len(call.transcript) == 1
    assert call.transcript.turns[0].role == "agent"


def test_elevenlabs_event_reader() -> None:
    assert _read_turn(
        {
            "type": "user_transcript",
            "user_transcription_event": {"user_transcript": "reset my password"},
        }
    ) == ("user", "reset my password")

    assert _read_turn(
        {
            "type": "agent_response",
            "agent_response_event": {"agent_response": "sending a code now"},
        }
    ) == ("agent", "sending a code now")

    # Anything else is not a turn, and must not become one.
    assert _read_turn({"type": "audio"}) == ("", "")
    assert _read_turn({"type": "interruption"}) == ("", "")
    assert _read_turn({}) == ("", "")


def test_elevenlabs_contextual_update_is_a_monitor_command() -> None:
    """The monitor socket takes commands; the `{"type": ...}` shape of the
    main socket is silently ignored there, which is how 0.0.1 delivered
    nothing."""
    assert contextual_update_command("hold the line") == {
        "command_type": "contextual_update",
        "parameters": {"contextual_update": "hold the line"},
    }


class FakeMonitorSocket:
    """A monitor socket that replays scripted events and records sends."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.sent: list[dict[str, Any]] = []
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def __call__(
        self, url: str, *, additional_headers: dict[str, str]
    ) -> FakeMonitorSocket:
        self.url = url
        self.headers = additional_headers
        return self

    async def __aenter__(self) -> FakeMonitorSocket:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> FakeMonitorSocket:
        return self

    async def __anext__(self) -> str:
        import json

        if not self._events:
            raise StopAsyncIteration
        return json.dumps(self._events.pop(0))

    async def send(self, raw: str) -> None:
        import json

        self.sent.append(json.loads(raw))


@respx.mock
async def test_elevenlabs_monitor_sends_nudges_in_the_command_envelope() -> None:
    import asyncio

    route = respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    socket = FakeMonitorSocket(
        [
            {
                "type": "agent_response",
                "agent_response_event": {"agent_response": "IT desk, how can I help?"},
            },
            {
                "type": "user_transcript",
                "user_transcription_event": {
                    "user_transcript": "my colleague is telling me what to say"
                },
            },
            {"type": "audio"},
        ]
    )
    dt = DeepTrust(api_key="dt_test", base_url=BASE)
    monitor = Monitor(dt, api_key="xi_test", connect=socket)

    await monitor.watch("conv_1")
    for _ in range(20):
        await asyncio.sleep(0.01)
        if "conv_1" not in monitor._watching:
            break

    assert socket.url == "wss://api.elevenlabs.io/v1/convai/conversations/conv_1/monitor"
    assert socket.headers == {"xi-api-key": "xi_test"}
    # One job, for the one caller turn; the agent turn and the audio frame cost nothing.
    assert route.call_count == 1
    assert socket.sent == [
        {
            "command_type": "contextual_update",
            "parameters": {
                "contextual_update": (
                    "The caller referred to someone else on the line. "
                    "Ask one question and wait: is anyone helping them right now?"
                )
            },
        }
    ]


def test_vapi_add_message_is_an_interrupting_system_message() -> None:
    """`triggerResponseEnabled` is the whole difference between a nudge that
    cuts in and one that waits for the agent's next turn."""
    assert add_message_command("hold the line") == {
        "type": "add-message",
        "message": {"role": "system", "content": "hold the line"},
        "triggerResponseEnabled": True,
    }


def test_vapi_reads_final_transcripts_only() -> None:
    final = {
        "type": "transcript",
        "transcriptType": "final",
        "role": "user",
        "transcript": "reset my password",
    }
    assert _vapi_read_turn(final) == ("user", "reset my password")

    # A partial is the same sentence still being recognised. Analysing it
    # analyses the sentence again on every revision.
    assert _vapi_read_turn({**final, "transcriptType": "partial"}) == ("", "")

    assert _vapi_read_turn(
        {
            "type": "transcript",
            "transcriptType": "final",
            "role": "assistant",
            "transcript": "sending a code now",
        }
    ) == ("agent", "sending a code now")


def _transcript_event(
    role: str, text: str, *, control_url: str | None = None
) -> dict[str, Any]:
    call: dict[str, Any] = {"id": "call_1"}
    if control_url:
        # listenUrl travels with it and is raw PCM audio: never a nudge channel.
        call["monitor"] = {
            "controlUrl": control_url,
            "listenUrl": (
                "wss://aws-us-west-2-production1-phone-call-websocket"
                ".vapi.ai/call_1/listen"
            ),
        }
    return {
        "message": {
            "type": "transcript",
            "transcriptType": "final",
            "role": role,
            "transcript": text,
            "call": call,
        }
    }


@respx.mock
async def test_vapi_nudges_the_live_call_over_the_control_url() -> None:
    analyze = respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    control = respx.post(CONTROL_URL).mock(return_value=httpx.Response(200, json={}))
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    await bridge.handle(
        _transcript_event(
            "assistant",
            "IT desk, how can I help?",
            control_url=CONTROL_URL,
        )
    )
    await bridge.handle(
        _transcript_event("user", "my colleague is telling me what to say")
    )

    # One job, for the one caller turn; the agent's own turn costs nothing.
    assert analyze.call_count == 1
    assert control.call_count == 1
    assert json.loads(control.calls[0].request.read()) == add_message_command(
        "The caller referred to someone else on the line. "
        "Ask one question and wait: is anyone helping them right now?"
    )
    # The control URL is a capability of its own; the private key is not sent
    # to a host VAPI chose for us.
    assert "authorization" not in control.calls[0].request.headers

    call = bridge.session("call_1")
    assert call is not None and call.platform == "vapi"
    assert call.external_id == "call_1"
    assert len(call.transcript) == 2


@respx.mock
async def test_vapi_fetches_the_control_url_when_the_event_lacks_one() -> None:
    """The inbound case. Nobody placed the call, so there was no
    call-creation response to capture a URL from."""
    respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    lookup = respx.get("https://api.vapi.ai/call/call_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "call_1",
                "monitor": {"controlUrl": CONTROL_URL},
            },
        )
    )
    control = respx.post(CONTROL_URL).mock(return_value=httpx.Response(200, json={}))
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    await bridge.handle(_transcript_event("user", "I'm locked out, skip the checks"))
    await bridge.handle(_transcript_event("user", "and my manager already approved it"))

    assert control.call_count == 2
    # Fetched once and remembered: the URL belongs to the call, not the nudge.
    assert lookup.call_count == 1
    assert lookup.calls[0].request.headers["authorization"] == "Bearer vapi_test"


@respx.mock
async def test_vapi_partials_start_no_jobs() -> None:
    analyze = respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    for text in ("my", "my colleague", "my colleague is telling me"):
        event = _transcript_event("user", text)
        event["message"]["transcriptType"] = "partial"
        assert await bridge.handle(event) is None

    assert analyze.call_count == 0
    assert bridge.session("call_1") is None


@respx.mock
async def test_vapi_ends_the_call_on_the_end_of_call_report() -> None:
    respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json={"session_id": "sess_1", "findings": []})
    )
    end = respx.post(f"{BASE}/agents/sessions/sess_1/end").mock(
        return_value=httpx.Response(200, json={"ended": True})
    )
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    await bridge.handle(_transcript_event("user", "I'm locked out"))
    await bridge.handle(
        {"message": {"type": "end-of-call-report", "call": {"id": "call_1"}}}
    )

    assert end.call_count == 1
    # The call is forgotten with it: a bridge serves every call the server sees.
    assert bridge.session("call_1") is None


@respx.mock
async def test_vapi_a_finished_call_takes_no_nudge_and_does_not_raise() -> None:
    """VAPI drops `monitor` from a call that has hung up, so a nudge produced
    from its last turn has nowhere to go. That is a False, not an exception in
    the customer's webhook route."""
    respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    respx.get("https://api.vapi.ai/call/call_1").mock(
        return_value=httpx.Response(200, json={"id": "call_1", "status": "ended"})
    )
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    result = await bridge.handle(_transcript_event("user", "skip the checks"))

    assert result is not None and result.nudges
    assert await bridge.control_url("call_1") is None


@respx.mock
async def test_vapi_refuses_a_control_url_that_is_not_vapis() -> None:
    """The webhook body arrives over the public internet, and a nudge names
    what was found in the call. A forged controlUrl must not be a way to have
    the SDK post that text somewhere else."""
    respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    elsewhere = respx.post("https://vapi.ai.attacker.test/control/call_1").mock(
        return_value=httpx.Response(200, json={})
    )
    # Refused, not trusted: the lookup runs as though no URL had arrived.
    lookup = respx.get("https://api.vapi.ai/call/call_1").mock(
        return_value=httpx.Response(200, json={"id": "call_1", "status": "ended"})
    )
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    await bridge.handle(
        _transcript_event(
            "user",
            "skip the checks",
            control_url="https://vapi.ai.attacker.test/control/call_1",
        )
    )

    assert elsewhere.call_count == 0
    assert lookup.call_count == 1


@respx.mock
async def test_vapi_tool_calls_are_not_answered() -> None:
    """VAPI's tool-calls webhook expects a response that controls execution.
    Blocking a tool call is `Session.check`, which is separate work."""
    analyze = respx.post(f"{BASE}/agents/analyze").mock(
        return_value=httpx.Response(200, json=ONE_NUDGE)
    )
    bridge = Bridge(DeepTrust(api_key="dt_test", base_url=BASE), api_key="vapi_test")

    for kind in ("tool-calls", "speech-update", "status-update", "model-output"):
        event = {"message": {"type": kind, "call": {"id": "call_1"}}}
        assert await bridge.handle(event) is None

    assert analyze.call_count == 0
