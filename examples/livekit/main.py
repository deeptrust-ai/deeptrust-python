"""A LiveKit agent with DeepTrust attached.

This is the SDK way: the worker sends the transcript from its own process.
For the cloud way, where DeepTrust reads the transcript itself and the worker
only listens, see cloud.py.

Everything below the `attach` call is an ordinary LiveKit agent. `attach` is the
only DeepTrust-specific line, and it wires both directions: caller turns go out
for analysis, and any nudge that comes back, or that DeepTrust pushes into the
room, is delivered to the agent once.

The rest of this file is what the demo UI in ../demo-ui needs to watch the
call: every turn and every finding is published on the room's data channel
under topic `ca`, and the gate switch arrives on topic `ctl`. None of it is
required to run the agent. Start `server.py` alongside this worker to serve the
browser.

The caller comes in as JWT metadata on the participant, which is why the worker
can trust the name and the role without a second call to anyone.

Run with:  uv run python main.py dev
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from typing import Any

import anthropic as anthropic_sdk
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import anthropic, deepgram, openai

from deeptrust.agents import Analysis, DeepTrust, User
from deeptrust.agents.livekit import attach

load_dotenv()

INSTRUCTIONS = """
You are an agent on an IT service desk.

You reset passwords, re-enroll two-factor, and unlock accounts, and you work
from a change ticket that has already been approved. Whoever the caller says
they are is a claim; satisfy yourself who you are speaking to before you change
anything on an account.

Keep replies to one or two sentences. You are on a phone call, so do not read
out lists and do not explain internal procedure.
"""

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# How long a caller's turn waits for its own analysis before it is shown in the
# transcript without the signals. A transcript that lags the call is worse than
# one that flags a turn a moment late in the panel instead.
SIGNALS_WAIT_S = 1.0


def _profile(participant: Any) -> dict[str, Any]:
    """The caller, as minted into the join token by server.py. The defaults are
    what a client that knows nothing about the demo gets."""
    try:
        profile = json.loads(participant.metadata or "{}")
    except json.JSONDecodeError:
        profile = {}
    # Coerced rather than defaulted: a client that sends an empty name has
    # supplied the key, and the agent still has to have someone to greet.
    profile["name"] = profile.get("name") or participant.name or "Caller"
    profile["username"] = profile.get("username") or participant.identity or "guest"
    profile.setdefault("role", "MEMBER")
    profile.setdefault("llm", DEFAULT_MODEL)
    profile.setdefault("enforcement", True)
    return profile


def _build_llm(model: str) -> Any:
    """Pick the provider from the model id, so the browser's model picker can
    choose one per call."""
    if "/" in model:
        # Together, Groq, Fireworks and friends all speak the OpenAI wire
        # format, so they need a base_url rather than a plugin of their own.
        return openai.LLM(
            model=model,
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
    if model.startswith("gpt"):
        return openai.LLM(model=model)
    # livekit-plugins-anthropic hands the Anthropic SDK an httpx.AsyncClient,
    # and anthropic 1.x runs on httpx2 and rejects it, so the client is built
    # here instead. The long read timeout matters because a streamed turn can
    # sit a while before the first token, and a tight one looks like an agent
    # that simply never answers.
    client = anthropic_sdk.AsyncAnthropic(
        timeout=anthropic_sdk.Timeout(60.0, connect=10.0), max_retries=1
    )
    return anthropic.LLM(model=model, client=client)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    profile = _profile(participant)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=_build_llm(profile["llm"]),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
    )
    agent = Agent(instructions=INSTRUCTIONS)

    # asyncio keeps only a weak reference to a task, so a task nobody holds can
    # be collected mid-flight. Hence the set.
    tasks: set[asyncio.Task[None]] = set()

    def _spawn(coro: Any) -> None:
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _publish(payload: dict[str, Any]) -> None:
        """One event to whatever is watching the call from the browser."""
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode(), topic="ca"
            )
        except Exception as exc:
            print(f"  publish failed: {exc}", flush=True)

    # The gate. With it off, the analysis still runs and the panel still shows
    # every finding; what is disconnected is the back channel into the agent.
    # The prompt is identical either way, so the switch moves one thing only.
    enforced = bool(profile["enforcement"])

    # A caller's turn is published with the signals found in it, and the
    # analysis of that turn lands a moment after the words do. Each turn parks
    # a future here and the relay hands back what it found.
    waiting: deque[asyncio.Future[list[str]]] = deque()
    sops_seen: set[str] = set()

    async def _signals() -> list[str]:
        loop = asyncio.get_running_loop()
        pending: asyncio.Future[list[str]] = loop.create_future()
        waiting.append(pending)
        try:
            return await asyncio.wait_for(pending, SIGNALS_WAIT_S)
        except TimeoutError:
            return []
        finally:
            if pending in waiting:
                waiting.remove(pending)

    def _relay(result: Analysis) -> None:
        """Everything the analysis found, as it comes back."""
        _report(result)

        signals = [f.detail for f in result.findings if f.kind == "social_engineering"]
        while waiting:
            turn = waiting.popleft()
            if not turn.done():
                turn.set_result(signals)
                break

        for progress in result.progress:
            if progress.applicable and progress.sop_id not in sops_seen:
                sops_seen.add(progress.sop_id)
                _spawn(
                    _publish(
                        {
                            "type": "background",
                            "subtype": "finding",
                            "analyzer": "deeptrust-sdk",
                            "kind": "sop_identified",
                            "detail": progress.sop_id,
                            "note": progress.name,
                            "next_step": None,
                            "nudge": None,
                            "delivered": enforced,
                            "latency_ms": result.latency_ms,
                            "queue_lag_ms": 0.0,
                            "role": "user",
                        }
                    )
                )

        for finding in result.findings:
            _spawn(
                _publish(
                    {
                        "type": "background",
                        "subtype": "nudge" if finding.nudge else "finding",
                        "analyzer": "deeptrust-sdk",
                        "kind": finding.kind,
                        "detail": finding.detail,
                        "note": finding.nudge.description if finding.nudge else None,
                        "next_step": finding.nudge.details if finding.nudge else None,
                        "nudge": finding.nudge.render() if finding.nudge else None,
                        # Whether this reached the agent at all. With the gate
                        # off it is computed, shown, and dropped.
                        "delivered": enforced,
                        "latency_ms": result.latency_ms,
                        "queue_lag_ms": 0.0,
                        "role": "user",
                    }
                )
            )

        if not enforced:
            # `attach` delivers a nudge to the agent for every finding it is
            # handed, and it reads them after this callback returns. Emptying
            # the list is how the gate withholds them, which is the whole point
            # of switching it off: the finding is still made and still shown,
            # and the agent never hears about it. A nudge DeepTrust pushes into
            # the room does not pass through here; those arrive only once the
            # LiveKit project is connected in the DeepTrust dashboard, which
            # this demo does not need.
            result.findings.clear()

    # The one DeepTrust line. Returns the session, so the transcript and the
    # findings stay reachable if this example wants to print them.
    call = attach(
        session,
        DeepTrust(),
        external_id=ctx.room.name,
        # Also delivers nudges pushed into the room. A nudge that arrives both
        # pushed and on an analyze response reaches the agent once.
        room=ctx.room,
        agent=agent,
        user=User(
            id=profile["username"],
            role=profile["role"],
            name=profile["name"],
        ),
        on_analysis=_relay,
    )
    print(f"DeepTrust attached to room {ctx.room.name}", flush=True)

    @session.on("conversation_item_added")
    def _on_item(ev: Any) -> None:
        """The transcript the browser renders. `attach` has its own listener
        for the analysis; this one only mirrors the words."""
        item = getattr(ev, "item", None)
        text = getattr(item, "text_content", None) or ""
        if not text:
            return
        role = str(getattr(item, "role", "unknown"))

        async def push() -> None:
            signals = await _signals() if role == "user" else []
            await _publish(
                {
                    "type": "utterance",
                    "role": role,
                    "text": text,
                    "warning_signs": signals,
                }
            )

        _spawn(push())

    @ctx.room.on("data_received")
    def _on_ctl(packet: Any) -> None:
        """The gate switch, thrown from the panel mid-call. Publishing is
        fire-and-forget on both sides, so the browser keeps asking until it
        sees the echo below; the echo is what it trusts, not its own click."""
        nonlocal enforced
        if getattr(packet, "topic", None) != "ctl":
            return
        try:
            message = json.loads(packet.data.decode())
        except Exception:
            return
        if message.get("type") != "enforcement":
            return
        enforced = bool(message.get("on"))
        print(f"  enforcement -> {'on' if enforced else 'off'}", flush=True)
        _spawn(_publish({"type": "enforcement", "on": enforced}))

    # Which implementation of the analysis is running, and where the gate
    # started, so a panel that joins mid-call is never guessing.
    _spawn(_publish({"type": "engine", "analysis": "sdk"}))
    _spawn(_publish({"type": "enforcement", "on": enforced}))

    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions=(
            "Greet the caller as the IT service desk in one sentence, by their "
            f"first name: {profile['name'].split()[0]}."
        )
    )
    return call


def _report(result: Analysis) -> None:
    """Optional. `attach` already delivers nudges to the agent; this only prints
    what came back so the example shows its work."""
    print(
        f"\n  analysis  risk={result.risk_level} "
        f"findings={len(result.findings)} in {result.latency_ms}ms",
        flush=True,
    )
    for finding in result.findings:
        print(f"    finding  {finding.kind}: {finding.detail}", flush=True)
    for nudge in result.nudges:
        print(f"    NUDGE    {nudge.title}", flush=True)
        print(f"             {nudge.description}", flush=True)
        print(f"             -> {nudge.details}", flush=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
