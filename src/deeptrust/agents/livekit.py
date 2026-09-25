"""LiveKit adapter.

There are two ways to run DeepTrust on LiveKit, and each is one call.

The cloud way. Connect the LiveKit project in the DeepTrust dashboard
(Settings, Voice Agents) and add the webhook it gives you in LiveKit Cloud.
DeepTrust then reads the call's transcript itself and pushes nudges into the
room. The worker only has to listen for them, and needs no DeepTrust key:

    from deeptrust.agents.livekit import listen

    listen(ctx.room, session, agent=agent)

The SDK way. The worker sends the transcript from this process and delivers the
nudges that come back:

    from deeptrust.agents import DeepTrust, User
    from deeptrust.agents.livekit import attach

    attach(
        session,
        DeepTrust(),
        external_id=ctx.room.name,
        room=ctx.room,
        user=User(id=account_id, role="MEMBER"),
    )

`attach` analyzes the transcript whenever the caller says something new, and
with `room` it also listens for pushed nudges. Either way a nudge reaches the
agent once, however many times and by whichever route it arrives.

A nudge is added to the agent's chat context, and by default also interrupts:
because the agent runs in this process, a nudge can arrive while it is still
generating and stop a reply part-way through. Pass `interrupt=False` to add the
nudge to the context and let the current reply finish.

Install with the extra:  pip install "deeptrust-ai[livekit]"
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any

from ..errors import DeepTrustError
from ..types import Nudge, User
from . import DeepTrust

# The data channel topic DeepTrust publishes nudges on, and the `type` inside
# the payload. Both are checked: the topic routes, the type confirms.
NUDGE_TOPIC = "deeptrust.nudge"


def _spawner() -> Callable[[Any], None]:
    # asyncio holds only a weak reference to a task, so a task nobody keeps can
    # be garbage collected mid-flight. Hence the set.
    tasks: set[asyncio.Task[None]] = set()

    def spawn(coro: Any) -> None:
        t = asyncio.create_task(coro)
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    return spawn


def _read_packet(packet: Any) -> Nudge | None:
    """The nudge in a data packet, or None if the packet is not one.

    Only packets sent from the server side are accepted. DeepTrust sends with
    the server API, which LiveKit delivers with no participant. Anything a
    participant publishes, including the caller's own client, could otherwise
    put words into the agent's context by using the same topic.
    """
    if getattr(packet, "topic", None) != NUDGE_TOPIC:
        return None
    if getattr(packet, "participant", None) is not None:
        return None
    try:
        d = json.loads(bytes(packet.data).decode("utf-8"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("type") != NUDGE_TOPIC:
        return None

    def _str(key: str) -> str:
        v = d.get(key)
        return v if isinstance(v, str) else ""

    nudge = Nudge(
        title=_str("title"),
        description=_str("description"),
        details=_str("details"),
        id=_str("id") or None,
    )
    if not nudge.render() and _str("text"):
        # `text` is the rendered form. A payload that carries only that is still
        # something to say, so it goes in as the description.
        nudge = Nudge(
            title=nudge.title, description=_str("text"), details="", id=nudge.id
        )
    return nudge if nudge.render() else None


class _Delivery:
    """Delivers nudges to one agent session, each one at most once."""

    def __init__(self, agent_session: Any, agent: Any, interrupt: bool) -> None:
        self._session = agent_session
        self._agent = agent
        self._interrupt = interrupt
        self._seen: set[str] = set()

    async def deliver(self, nudge: Nudge) -> None:
        text = nudge.render()
        if not text:
            return
        # The push and the analyze response carry the same id for the same
        # nudge. An older API sends no id, and then the text is the only thing
        # the two have in common, so both are recorded and either one matches.
        keys = {f"text:{text}"}
        if nudge.id:
            keys.add(f"id:{nudge.id}")
        if keys & self._seen:
            return
        target = self._agent or getattr(self._session, "current_agent", None)
        if target is None:
            return
        # Marked before the first await, so a push and a response that arrive
        # together cannot both get past the check.
        self._seen |= keys

        # Delivered once, as a system message in the chat context. A message
        # there stays for the rest of the call and shapes every later reply,
        # where `instructions` apply to one reply only. The reply generated to
        # interrupt then reads it from the context, so passing the text again as
        # instructions would only say the same thing twice.
        chat = target.chat_ctx.copy()
        chat.add_message(role="system", content=text)
        await target.update_chat_ctx(chat)
        if self._interrupt:
            self._session.interrupt()
            self._session.generate_reply()

    def listen(self, room: Any, spawn: Callable[[Any], None]) -> Callable[[], None]:
        """Deliver nudges pushed into `room`. Returns a function that stops."""

        def _on_data(packet: Any) -> None:
            nudge = _read_packet(packet)
            if nudge is not None:
                spawn(self.deliver(nudge))

        room.on("data_received", _on_data)
        stopped = False

        def stop() -> None:
            nonlocal stopped
            if not stopped:
                stopped = True
                room.off("data_received", _on_data)

        return stop


def listen(
    room: Any,
    agent_session: Any,
    *,
    agent: Any = None,
    interrupt: bool = True,
) -> Callable[[], None]:
    """Deliver the nudges DeepTrust pushes into `room` to the agent.

    For the cloud way, where DeepTrust reads the transcript itself: this never
    calls the DeepTrust API and needs no key. `room` is the `livekit.rtc.Room`
    the agent is in, usually `ctx.room`.

    Pass `agent` if it is available. Otherwise it is read from
    `agent_session.current_agent`.

    Returns a function that stops listening. It stops by itself when the
    session closes.
    """
    delivery = _Delivery(agent_session, agent, interrupt)
    stop = delivery.listen(room, _spawner())
    agent_session.on("close", lambda _ev: stop())
    return stop


def attach(
    agent_session: Any,
    dt: DeepTrust,
    *,
    external_id: str,
    room: Any = None,
    agent: Any = None,
    user: User | None = None,
    interrupt: bool = True,
    on_analysis: Callable[[Any], None] | None = None,
) -> Any:
    """Wire a LiveKit AgentSession to DeepTrust. Returns the DeepTrust session.

    Analysis runs on caller turns only. Feeding an agent's own replies back in
    doubles the work and lets its answers reclassify the call.

    Pass `room`, usually `ctx.room`, to also deliver nudges DeepTrust pushes
    into the room. A nudge that arrives both ways is delivered once.

    Pass `agent` if it is available. Otherwise it is read from
    `agent_session.current_agent`.

    When the session closes, the DeepTrust call is ended so post-call
    processing starts at once.

    Returns the DeepTrust session, so the transcript and findings remain
    reachable.
    """
    call = dt.session(external_id=external_id, user=user, platform="livekit")
    spawn = _spawner()
    delivery = _Delivery(agent_session, agent, interrupt)
    stop = delivery.listen(room, spawn) if room is not None else None

    async def _run(role: str, text: str) -> None:
        call.append(role, text)
        if role != "user":
            return
        result = await call.analyze()
        if result is None:
            return
        if on_analysis:
            on_analysis(result)
        for nudge in result.nudges:
            await delivery.deliver(nudge)

    # The last turn taken, so a repeat of it can be recognised. LiveKit emits a
    # conversation item more than once for the same speech in practice, and a
    # transcript that carries the same sentence twice is analysed twice.
    last: dict[str, str] = {}

    def _on_item(ev: Any) -> None:
        item = getattr(ev, "item", None)
        text = getattr(item, "text_content", None) or ""
        if not text:
            return
        role = "user" if str(getattr(item, "role", "unknown")) == "user" else "agent"

        # Compared against the previous turn only, not the whole transcript: a
        # caller who says the same thing again later in the call means it, and
        # that repetition is itself worth analysing.
        if last.get(role) == text:
            return
        last[role] = text

        spawn(_run(role, text))

    async def _end() -> None:
        # The server ends an idle call on its own a few minutes later, so a
        # failed end costs time, not the record.
        with contextlib.suppress(DeepTrustError):
            await call.end()

    def _on_close(_ev: Any) -> None:
        if stop is not None:
            stop()
        spawn(_end())

    agent_session.on("conversation_item_added", _on_item)
    agent_session.on("close", _on_close)

    return call
