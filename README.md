<h1 align="center">Dreamer - self-evolving context for your coding agents</h1>
<!--<p align="center"><em>Self-evolving context for your coding agents.</em></p>-->

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/18d426c57df26e5b1e99727a3aca643d/raw/b2a312ad81c6b3fa05d1793a571b71e2de1feb5a/dreamer-light.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/18d426c57df26e5b1e99727a3aca643d/raw/b2a312ad81c6b3fa05d1793a571b71e2de1feb5a/dreamer-dark.svg">
  <img alt="Image" src="https://gist.githubusercontent.com/OKUA1/18d426c57df26e5b1e99727a3aca643d/raw/b2a312ad81c6b3fa05d1793a571b71e2de1feb5a/dreamer-light.svg">
</picture>

<p align="center">
  <a href="#get-started">Get started</a> ·
  <a href="#extensions">Extensions</a> ·
  <a href="https://luml.ai/blog/2026/dreamer-self-evolving-agents">Blogpost</a>
</p>

Dreamer keeps your team's `AGENTS.md` and skills up to date with what your
coding agents learn while they work. It runs as a self-hostable MCP server
that collects memories from every agent on the team and, on a schedule,
regenerates the context bundle the next session reads.

**Team-wide memory.** Memories from every agent on the team pool into a
single store and feed a single context bundle, instead of staying on one
workstation.

**Any coding CLI.** Anything that speaks MCP submits memories through the
same `submit_memory` tool, including Claude Code, Cursor, Codex, and custom
agents.

**Extendible by config.** STM store, LTM store, context store, dream engine,
auth, triggers, and hooks are Python `Protocol`s wired up from YAML. Swap
any default by pointing at a different class.


<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Dataforce-Solutions/static_assets/blob/main/CTA-Compact-Light.png?raw=true" >
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Dataforce-Solutions/static_assets/blob/main/CTA-Compact-Dark.png?raw=true">
  <img alt="Image" src="https://github.com/Dataforce-Solutions/static_assets/blob/main/CTA-Compact-Light.png?raw=true">
</picture>

## Get started

Dreamer requires Python 3.12 or later. The `defaults` extra pulls in SQLite
for STM, the Claude Agent SDK for dreaming, APScheduler for cron triggers,
and gitpython for the post-dream commit hook.

```bash
pip install 'dreamer-server[defaults]'
```

Scaffold a project.

```bash
dreamer init
```

This writes a `dreamer.yaml`, a `workspace/` with `memory/` and `context/`
subdirectories, and a `.gitignore` that keeps the SQLite database out of
git.

Issue a token for your agents to send in the `Authorization` header.

```bash
dreamer-simple-auth token create --db ./dreamer.db --name my-token
```

Sanity-check the config. The loader resolves every component, runs the
protocol-conformance check, and prints the wired graph and per-slot
multi-tenancy table.

```bash
dreamer config check
```

Run the server.

```bash
dreamer serve
```

Point Claude Code or any MCP client over streamable-http at
`http://localhost:8080/mcp/` with `Authorization: Bearer <token>`. The
server advertises a `submit_memory` tool whose accepted types come from your
config. Out of the box, those are `observation`, `failure`, and
`code_snippet`.

Cron is the default trigger. To fire a one-shot dream from the command
line:

```bash
dreamer dream --trigger external
```

## Extensions

Dreamer is config-assembled. `dreamer.yaml` wires `module.path.ClassName`
references into a component graph. Every slot sits behind a Python
`Protocol` defined in `dreamer.api`, including the STM store, the LTM
store, the context store, the dream engine, auth, triggers, and hooks. The
shipped defaults are chosen to get a team running in a few minutes, and
every one of them can be swapped.

```yaml
stm_store:
  class: dreamer.contrib.stm.sqlite.SQLiteSTMStore
  params:
    path: ./data/stm.db

ltm_store:
  class: dreamer.contrib.ltm.markdown.MarkdownLTMStore
  params:
    root: ./workspace/memory

context_store:
  class: dreamer.contrib.context.markdown.MarkdownContextStore
  params:
    root: ./workspace/context

dream_engine:
  class: dreamer.contrib.dream.claude_agent.ClaudeAgentDreamEngine

triggers:
  - class: dreamer.contrib.triggers.cron.CronTrigger
    params:
      schedule: "0 */6 * * *"
```

To plug in your own component, write a class that satisfies the protocol.
For example, a Postgres-backed STM store:

```python
from typing import ClassVar
from dreamer.api.compat import implements
from dreamer.api.stores import STMStore

@implements(STMStore, version=1)
class PostgresSTMStore:
    multi_tenant: ClassVar[bool] = True

    def __init__(self, *, dsn: str) -> None:
        ...

    async def submit(self, memory, *, ctx): ...
    async def claim_batch(self, *, ctx): ...
```

Then reference it from `dreamer.yaml`:

```yaml
stm_store:
  class: my_pkg.stores.PostgresSTMStore
  params:
    dsn: ${env:POSTGRES_DSN}
```

`dreamer config check` validates the protocol version, signatures,
parameter kinds, and capability requirements before the server boots. The
same shape of change covers a graph-backed long-term memory store, an OIDC
auth backend, or a Slack notification hook in place of the git commit.

`dreamer.testing.conformance` ships abstract `pytest` classes for each
protocol. The cases cover idempotency, lease isolation, expired-lease
reclamation, tenant-scope leakage, and the `purge_consumed` contract. Any
compliant implementation should pass them.

```python
from dreamer.testing.conformance.stm_store import STMStoreConformance

class TestPostgresSTMStore(STMStoreConformance):
    @pytest.fixture
    async def store(self):
        return PostgresSTMStore(dsn="postgresql://...")
```

## License

MIT. See [LICENSE](LICENSE).

<hr>
Dreamer is part of the broader <a href="https://github.com/luml-ai/luml">LUML</a> effort to build open infrastructure for autonomous ML agents. <br/><br/>

<div align="center">

<a href="https://github.com/luml-ai/luml">
  <img src="https://raw.githubusercontent.com/gist/OKUA1/bae081e15b6efae41cc42c7f88c8c2a2/raw/8e82db4bbba01f768176560ce6f039113ebd3d30/core.svg" alt="Core — registry, deployments, monitoring" width="270"/>
</a>
<a href="https://github.com/luml-ai/luml">
  <img src="https://gist.githubusercontent.com/OKUA1/bae081e15b6efae41cc42c7f88c8c2a2/raw/8e82db4bbba01f768176560ce6f039113ebd3d30/prisma.svg" alt="Prisma — autonomous ML research agents" width="270"/>
</a>
<a href="https://github.com/luml-ai/luml">
  <img src="https://raw.githubusercontent.com/gist/OKUA1/bae081e15b6efae41cc42c7f88c8c2a2/raw/8e82db4bbba01f768176560ce6f039113ebd3d30/flow.svg" alt="Flow — experiment tracking and tracing" width="270"/>
</a>

</div> 
