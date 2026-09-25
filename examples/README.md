# Examples

Each folder is a self-contained project with its own `pyproject.toml` and a
README showing real output from a real run.

| | |
|---|---|
| [`livekit/`](livekit) | An agent with DeepTrust attached in one line, the SDK way or the cloud way. Nudges can interrupt a reply in progress. |
| [`elevenlabs/`](elevenlabs) | An agent it provisions for you, watched from outside. No code inside the agent at all. |
| [`elevenlabs-webhook/`](elevenlabs-webhook) | The call-start webhook you already own, turned into a watched call. Your process holds the socket. |

All three resolve `deeptrust-ai` from this checkout rather than the published
package, so they exercise the code in this repo:

```toml
[tool.uv.sources]
deeptrust-ai = { path = "../..", editable = true }
```

They all read `DEEPTRUST_BASE_URL`, so `just devserver` in the repo root is
enough to run any of them end to end with no DeepTrust key and no network.

## Watching nudges arrive (DeepTrust developers only)

Nothing in this section is needed to run either example, and nothing a customer
runs uses it.

Analysis finishes in a worker on the backend, after the `analyze()` call that
dispatched it has already returned. So on a real call the nudges are produced
*after* the response that would have carried them, and the demo panel shows
nothing. That is not a problem in production: **delivery is the backend's job.**
It holds the organisation's own platform credential and pushes the nudge into
the LiveKit room or the ElevenLabs conversation itself, so an agent is nudged
whether or not a browser is watching, and the integrating developer writes
nothing for it beyond the one `attach(...)` line.

For local work on the analysis pipeline it is still useful to see nudges land
in the panel. `dev_nudge_db.py` does that by polling the backend's
`notifications` table and putting each new row on the SSE stream the panel
already reads. It is off unless you point it at a database:

```sh
export DEEPTRUST_DEV_DB_URL=postgresql://user:pass@127.0.0.1:54322/postgres
```

With that unset — which is the default, and the case for anyone without a
DeepTrust database — the poller never starts, nothing is imported, and both
examples behave exactly as they do today. It also needs `asyncpg`, which the
examples deliberately do not depend on; without it the poller declines to start
and says so once, rather than making everyone install a database driver to run
a demo that queries no database.
