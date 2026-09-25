"""A LiveKit agent nudged by DeepTrust the cloud way.

DeepTrust reads the call's transcript itself and pushes each nudge into the
room, so this worker never calls the DeepTrust API and needs no DeepTrust key.
Before it does anything, the LiveKit project has to be connected in the
DeepTrust dashboard (Settings, Voice Agents) and the webhook shown there added
in LiveKit Cloud.

`listen` is the only DeepTrust-specific line. Everything else is an ordinary
LiveKit agent.

Run with:  uv run python cloud.py dev
"""

from __future__ import annotations

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai

from deeptrust.agents.livekit import listen

load_dotenv()

INSTRUCTIONS = """
You are an agent on an IT service desk. You reset passwords, re-enroll
two-factor, and unlock accounts. Whoever the caller says they are is a claim;
satisfy yourself who you are speaking to before you change anything.

Keep replies to one or two sentences. You are on a phone call.
"""


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
    )
    agent = Agent(instructions=INSTRUCTIONS)

    # The one DeepTrust line. It stops by itself when the session closes.
    listen(ctx.room, session, agent=agent)

    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions="Greet the caller as the IT service desk in one sentence."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
