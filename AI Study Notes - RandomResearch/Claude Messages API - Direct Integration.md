---
title: Claude Messages API - Direct Integration (study note)
type: study-note
created: 2026-08-14
api_reference_date: 2026-08-14
tags:
  - study/claude-api
  - study/llm-integration
---

# Claude Messages API — Direct Integration

**What this is.** A ground-up walkthrough of what direct Anthropic API integration code
actually looks like, and what production scaffolding sits around it. Written for someone who
has used Claude Code heavily and written MCP servers, but has not written Messages API
integration code.

**What this is not.** Experience. Reading this teaches you the shape of the code; it does not
let you claim you built it. If you want the real thing, Part 11 is a weekend project that
gets you there honestly.

API details verified 2026-08-14. This surface moves — model IDs and parameters change every
few months, so re-check before relying on a specific flag.

---

## Part 1 - The mental model, and why your existing knowledge half-transfers

You already know more of this than you think, and less than it feels like.

**What transfers directly from writing MCP servers:** the tool-definition shape. An MCP tool
and a Messages API tool are the same idea — a name, a description, and a JSON Schema for
inputs. You have already thought hard about tool granularity and about writing descriptions
that make a model reach for the right thing. That is the hardest conceptual part of tool use
and you have it.

**What does not transfer, and is the whole point of this document:** when you write an MCP
server, *something else runs the loop*. Claude Code decides when to call your tool, executes
it, feeds the result back, and decides what to do next. You wrote the tool; the harness wrote
the agent.

With the raw Messages API there is no harness. `POST /v1/messages` is a **single stateless
function call**: you send the whole conversation, you get one response back, and the
connection closes. If the model wants to use a tool, the response says so and *stops*. You
run the tool. You append the result. You call the API again. That `while` loop is yours to
write, and every agentic behavior you take for granted in Claude Code — multi-turn tool use,
context management, retries, cost accounting — is scaffolding somebody wrote around that
loop.

The whole rest of this document is that scaffolding, layer by layer.

**One clarifying fact:** the API is **stateless**. There is no conversation ID, no server-side
session. If you want turn 12 to remember turn 1, you send turns 1 through 11 with every
request. This is why prompt caching (Part 8) matters so much — without it you re-pay full
price for the entire history on every single turn.

---

## Part 2 - The minimal call

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=4096,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
```

Three things to notice, because each one bites people:

**`content` is a list of typed blocks, not a string.** A response can contain `text`,
`thinking`, `tool_use`, and several server-tool result types, in order. `response.content[0].text`
works right up until the model thinks first, at which point block 0 is a `thinking` block and
you get an `AttributeError` in production. Always branch on `block.type`.

**`max_tokens` is required and it is a hard ceiling on everything the model emits** —
thinking *plus* visible text. Undersize it and you get truncation with
`stop_reason == "max_tokens"` and no error. Sane defaults: **~16000 non-streaming**, **~64000
when streaming**. Non-streaming requests with very large `max_tokens` risk HTTP timeouts,
which is why the ceiling is lower there.

**Model IDs are exact strings with no date suffix** (current generation):

| Model | ID | Input $/1M | Output $/1M | Context |
|---|---|---|---|---|
| Opus 5 | `claude-opus-5` | $5 | $25 | 1M |
| Sonnet 5 | `claude-sonnet-5` | $3 | $15 | 1M |
| Haiku 4.5 | `claude-haiku-4-5` | $1 | $5 | 200K |

Guessing an ID (`claude-opus-5-20260301`) is a 404. There is a Models API
(`client.models.retrieve(id)`) that returns live context windows and capability flags — use it
rather than hardcoding a table like this one, which will be stale in a quarter.

### Parameters that were removed, and will 400

If you learned this API from 2024-era material, three habits are now hard errors on current
models:

- **`temperature`, `top_p`, `top_k`** — removed on Opus 5, Opus 4.8/4.7, Sonnet 5, Fable 5.
  Steer with prompting instead. (`temperature=0` never guaranteed determinism anyway.)
- **`thinking={"type": "enabled", "budget_tokens": N}`** — removed. Use
  `thinking={"type": "adaptive"}` and control depth with `output_config={"effort": ...}`.
- **Assistant-turn prefill** (ending `messages` with a partial `{"role": "assistant", ...}` to
  force output shape) — 400s. Use structured outputs instead (Part 10).

### Thinking and effort

```python
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive", "display": "summarized"},
    output_config={"effort": "high"},   # low | medium | high | xhigh | max
    messages=[...],
)
```

- **Thinking is ON by default on Opus 5.** Omitting the parameter runs adaptive thinking.
  (This differs from Opus 4.8/4.7, where omitting it meant no thinking — a real migration
  trap, because a `max_tokens` sized tightly around the answer now truncates.)
- **`display` defaults to `"omitted"`** — thinking blocks arrive with empty text. Set
  `"summarized"` if you surface reasoning to a user; otherwise a streaming UI shows a long
  pause before anything appears. Raw chain of thought is never returned on any model.
- **`effort` lives inside `output_config`**, not top-level. It is the main
  intelligence/latency/cost dial. `high` is the default; `xhigh` for hard coding and agentic
  work; `low`/`medium` are genuinely strong on Opus 5 and are where the cost savings are.

---

## Part 3 - Multi-turn: you own the history

Because the API is stateless, a conversation is just a list you keep appending to.

```python
class Conversation:
    def __init__(self, client: anthropic.Anthropic, model: str, system: str | None = None):
        self.client = client
        self.model = model
        self.system = system
        self.messages: list[dict] = []

    def send(self, user_text: str, max_tokens: int = 8192) -> str:
        self.messages.append({"role": "user", "content": user_text})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            messages=self.messages,
        )

        # Append the full content list, not just the text. Dropping blocks
        # breaks thinking-block replay and tool-use continuity.
        self.messages.append({"role": "assistant", "content": response.content})
        return next((b.text for b in response.content if b.type == "text"), "")
```

Rules worth internalizing:

- First message must be `user`.
- Consecutive same-role messages are allowed — the API merges them into one turn.
- **Append `response.content` (the list), not the extracted string.** This is the single most
  common multi-turn bug. If you flatten to text you lose `tool_use` blocks and thinking
  blocks, and the next turn either errors or silently loses continuity.
- Thinking blocks must be echoed back **unchanged** when continuing on the same model. Read
  them if you like; do not edit or reconstruct them.

**The cost consequence.** Turn 20 of a conversation sends turns 1–19 as input. Input tokens
grow quadratically over a session. Without caching, a 50-turn conversation costs far more than
50× a single turn. This is not a detail — it is the dominant cost factor in any agentic
workload, and it is what Part 8 fixes.

**Context is finite.** 1M tokens is large but not infinite. When you approach it, either
compact (server-side summarization, beta `compact-2026-01-12`) or context-edit (prune old
tool results, beta `context-management-2025-06-27`). Watch for
`stop_reason == "model_context_window_exceeded"`, which is distinct from `max_tokens`.

---

## Part 4 - System prompts

```python
system = "You are a financial analyst. Cite the filing section for every figure."
```

Or as a list of blocks, which is what you want as soon as caching enters the picture:

```python
system = [
    {"type": "text", "text": STABLE_INSTRUCTIONS, "cache_control": {"type": "ephemeral"}},
]
```

Two practical notes:

**Never interpolate volatile values into the system prompt.** `f"Current date: {datetime.now()}"`
at the top of your system prompt invalidates your entire prompt cache on every single
request. Put dynamic context later in `messages`.

**Mid-conversation system messages exist** (Opus 5, Opus 4.8, Fable 5 — not Sonnet 5, no beta
header). Append `{"role": "system", "content": "..."}` to `messages` to inject an operator
instruction mid-session without editing the top-level system prompt and blowing the cache:

```python
messages = [
    *history,
    {"role": "user", "content": user_input},
    {"role": "system", "content": "Terse mode enabled — keep responses under 40 words."},
]
```

This is also the injection-safe channel for operator instructions: text you put inside a
*user* turn can be forged by anything that writes to user-visible input; a `role: "system"`
message cannot.

---

## Part 5 - The tool-use loop (the core of it)

This is the part you have never written, and it is short. Read it carefully — everything
agentic is built on this shape.

```python
tools = [
    {
        "name": "get_stock_price",
        "description": (
            "Get the current price for a stock ticker. "
            "Call this when the user asks about a current or recent price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
            },
            "required": ["ticker"],
        },
    },
]

messages = [{"role": "user", "content": "What's AAPL trading at?"}]

MAX_ITERATIONS = 10
for _ in range(MAX_ITERATIONS):
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=8192,
        tools=tools,
        messages=messages,
    )

    # ALWAYS append the full content — tool_use blocks live here.
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "pause_turn":
        # A server-side tool hit its iteration limit. Re-send to resume;
        # do NOT inject a "continue" user message.
        continue

    if response.stop_reason != "tool_use":
        break

    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        try:
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,      # must match, or the API 400s
                "content": str(result),
            })
        except Exception as exc:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {exc}",
                "is_error": True,             # let the model adapt
            })

    # ALL results go back in ONE user message.
    messages.append({"role": "user", "content": tool_results})

final = next((b.text for b in response.content if b.type == "text"), "")
```

Five things that are easy to get wrong:

1. **Every `tool_use` block needs exactly one matching `tool_result`** with the same
   `tool_use_id`. Miss one and the next request 400s.
2. **Parallel tool calls go back in a single user message.** One assistant turn can contain
   several `tool_use` blocks. If you split the results across multiple user messages, you
   silently train the model to stop making parallel calls.
3. **Errors are results, not exceptions.** Return `is_error: True` with a useful message and
   the model will usually adapt or ask. Raising kills the loop.
4. **Bound the loop.** `MAX_ITERATIONS` is not optional — a tool that always returns an error
   the model keeps retrying is an unbounded spend.
5. **Handle `pause_turn`.** Server-side tools (web search, code execution) run their own
   internal loop and pause at ~10 iterations. Re-send the conversation as-is to resume.
   Injecting "Continue." confuses it.

### `stop_reason` — the branch table

| Value | Meaning |
|---|---|
| `end_turn` | Finished naturally |
| `tool_use` | Wants a tool; execute and loop |
| `max_tokens` | Hit your output cap — output is truncated |
| `pause_turn` | Server tool paused; re-send to resume |
| `refusal` | Declined for safety; `content` may be empty or partial |
| `stop_sequence` | Hit a custom stop sequence |

**Check `stop_reason` before reading `content`.** On a `refusal`, `content` can be an empty
list — code that indexes `content[0]` unconditionally crashes. Also on Opus 5 and Fable 5,
safety classifiers can decline a request that returns HTTP 200; a `fallbacks` parameter
(beta) can re-run it server-side on another model.

### The Tool Runner — the loop, prewritten

The SDK ships a beta helper that writes that loop for you:

```python
from anthropic import beta_tool

@beta_tool
def get_stock_price(ticker: str) -> str:
    """Get the current price for a stock ticker.

    Args:
        ticker: Ticker symbol, e.g. AAPL.
    """
    return f"{ticker} is at $214.30"

runner = client.beta.messages.tool_runner(
    model="claude-opus-5",
    max_tokens=8192,
    tools=[get_stock_price],
    messages=[{"role": "user", "content": "What's AAPL trading at?"}],
)

for message in runner:
    print(message)
```

The schema is derived from the type hints and docstring. Approval gates, result inspection,
and per-turn intervention are all supported through hooks — "I need control" is usually not a
reason to hand-write the loop.

**But write the manual loop once first.** The runner hides exactly the mechanic you are trying
to learn, and the manual version is the thing you will be asked about in an interview.

---

## Part 6 - Streaming

Non-streaming with a large `max_tokens` will eventually hit an HTTP timeout — the SDK actively
refuses requests it estimates will exceed ~10 minutes. Streaming is the answer, and it is also
what any interactive product needs.

```python
with client.messages.stream(
    model="claude-opus-5",
    max_tokens=32000,
    messages=[{"role": "user", "content": "Write a long analysis of..."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    final = stream.get_final_message()   # full Message object
    print(f"\n\ntokens: {final.usage.output_tokens}")
```

`text_stream` is the convenience path. For finer control — separating thinking deltas from
text deltas, or streaming tool inputs — iterate raw events:

```python
with client.messages.stream(..., thinking={"type": "adaptive", "display": "summarized"}) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
```

Event types, in order: `message_start` → `content_block_start` → `content_block_delta`* →
`content_block_stop` → `message_delta` (carries `stop_reason` and usage) → `message_stop`.

`get_final_message()` is the important part — it gives you the fully accumulated `Message`, so
you can stream to a UI *and* inspect `stop_reason` / `usage` / tool blocks for the loop, without
hand-assembling deltas.

---

## Part 7 - Token counting

```python
count = client.messages.count_tokens(
    model="claude-opus-5",
    system=system,
    messages=messages,
    tools=tools,
)
print(count.input_tokens)
```

**Do not use `tiktoken`.** It is OpenAI's tokenizer. It undercounts Claude tokens by roughly
15–20% on prose and much worse on code — and Claude's tokenizer changed with Opus 4.7, so
counts differ *between Claude generations* too. Token counts are model-specific: pass the same
model ID you will use for inference.

Use it for: pre-flight cost estimation, deciding when to compact, and re-baselining after a
model migration (never apply a blanket multiplier).

---

## Part 8 - Prompt caching (the highest-leverage thing here)

Cache reads cost ~0.1× base input price. Writes cost 1.25× (5-min TTL) or 2× (1-hour). On any
multi-turn or shared-prefix workload this is the difference between viable and not.

**The one invariant everything follows from: caching is a prefix match. Any byte change
anywhere in the prefix invalidates everything after it.**

Render order is **`tools` → `system` → `messages`**. A breakpoint on the last system block
caches tools and system together.

```python
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=8192,
    system=[
        {"type": "text", "text": LARGE_STABLE_PROMPT,
         "cache_control": {"type": "ephemeral"}},
    ],
    messages=[{"role": "user", "content": question}],
)

print(response.usage.cache_creation_input_tokens)  # written (paid ~1.25x)
print(response.usage.cache_read_input_tokens)      # read (paid ~0.1x)
print(response.usage.input_tokens)                 # uncached remainder (full price)
```

Note that `input_tokens` is only the *uncached remainder*. Total prompt size is the sum of all
three fields — a long agentic run showing `input_tokens: 4000` may have read a million cached
tokens.

### Placement

- **Large shared system prompt:** breakpoint on the last system block.
- **Multi-turn conversation:** breakpoint on the last content block of the most recent turn.
  Each request then reuses the whole prior conversation. Earlier breakpoints stay valid as
  read points, so hits accrue as the conversation grows.
- **Shared prefix, varying question:** breakpoint at the end of the *shared* part — not the end
  of the whole prompt. Otherwise every request writes a distinct entry and nothing is ever read.

Max **4** breakpoints per request. Minimum cacheable prefix is **512 tokens on Opus 5** (1024
on Opus 4.8, 4096 on Opus 4.6 / Haiku 4.5). Below the minimum it silently does not cache — no
error, just `cache_creation_input_tokens: 0`.

### Silent invalidators — grep for these

| Pattern | Why it kills the cache |
|---|---|
| `datetime.now()` in the system prompt | Prefix differs every request |
| `uuid4()` / request IDs early in content | Same |
| `json.dumps(d)` without `sort_keys=True` | Non-deterministic key order → different bytes |
| Session/user ID interpolated into system | Per-user prefix, no sharing |
| `if flag: system += ...` | Every flag combination is a distinct prefix |
| `tools=build_tools(user)` varying per user | Tools render at position 0 — nothing caches |

**Debug rule:** if `cache_read_input_tokens` is 0 across repeated identical-prefix requests,
you have a silent invalidator. Diff the rendered prompt bytes between two requests.

### Two more traps

**Changing tools or model mid-conversation invalidates everything.** Tools render at position
0. Serialize them deterministically (sort by name). Caches are model-scoped, so switching
models mid-session starts cold.

**Concurrent requests all miss.** A cache entry becomes readable only once the first response
*begins streaming*. Firing N identical-prefix requests in parallel means all N pay full price.
For fan-out: send one, await the first token, then fire the rest.

---

## Part 9 - Errors and retries

```python
import anthropic

try:
    response = client.messages.create(...)
except anthropic.NotFoundError:
    ...   # 404 — bad model ID
except anthropic.RateLimitError as e:
    retry_after = int(e.response.headers.get("retry-after", "60"))
except anthropic.APIStatusError as e:
    if e.status_code >= 500:
        ...   # retryable
    else:
        ...   # 4xx — fix the request
except anthropic.APIConnectionError:
    ...   # network failure before any response
```

Catch a **chain, most-specific first**, not one broad class. A single
`except APIStatusError` throws away the distinction between retryable (429, 5xx, network) and
non-retryable (400, 404) — which is the only thing your retry logic cares about.

**The SDK already retries.** `max_retries` defaults to 2, covering 408/409/429/5xx and
connection errors with exponential backoff. Default timeout is 10 minutes. Only hand-roll
backoff if you need behavior beyond that.

```python
client = anthropic.Anthropic(max_retries=5, timeout=30.0)

# Or per-request, without mutating the client:
client.with_options(timeout=5.0, max_retries=0).messages.create(...)
```

Note that timeouts are themselves retried, so worst-case wall clock is
`timeout × (max_retries + 1)`. A 10-minute timeout with 2 retries can block for 30 minutes.

**Log `response._request_id`.** Despite the underscore it is public, and it is what Anthropic
support needs to trace a failure.

The error table worth memorizing: **400** malformed request · **401** bad key · **403** no
permission · **404** bad model/endpoint · **413** too large · **429** rate limited (retryable)
· **500** server error (retryable) · **529** overloaded (retryable).

---

## Part 10 - Structured outputs

The 2024 way to force JSON was assistant prefill plus stop sequences plus a regex-and-retry
loop. Prefill now 400s, and the whole scaffold is replaced by one parameter.

```python
from pydantic import BaseModel

class Extraction(BaseModel):
    company: str
    fiscal_year: int
    revenue_usd: float
    segments: list[str]

response = client.messages.parse(
    model="claude-opus-5",
    max_tokens=4096,
    messages=[{"role": "user", "content": f"Extract from this filing:\n{text}"}],
    output_format=Extraction,
)

record = response.parsed_output   # a validated Extraction instance
```

`messages.parse()` derives the schema, constrains generation, and validates the result. The
raw form is `output_config={"format": {"type": "json_schema", "schema": {...}}}` — note
`output_config.format`, not the deprecated top-level `output_format` on `messages.create()`.

**Schema limits worth knowing:** `additionalProperties: false` is required on every object.
Recursive schemas are not supported. Numeric constraints (`minimum`, `maximum`) and string
length constraints are not enforced server-side — the Python and TypeScript SDKs strip them
from the schema and validate client-side instead. Incompatible with citations (400).

**Still check `stop_reason`.** A `refusal` or a `max_tokens` truncation means the output does
not match your schema regardless of the constraint.

For guaranteeing *tool input* shape rather than response shape, there is a separate mechanism:
`strict: True` on the tool definition, with `additionalProperties: false` and `required` in the
schema.

---

## Part 11 - What production scaffolding actually looks like

The API calls above are maybe 15% of a real integration. Here is the rest, ordered by how
early you will need it.

**1. Cost and token accounting — build this first.** Without per-call usage logging, every
other optimization is invisible. Log, per request: model, `input_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`, `stop_reason`,
latency, `_request_id`, and a route or feature label. Then compute cost from the current rate
card. You cannot tune caching or effort without this, and the first thing it usually reveals is
that cache hit rate is zero because of a `datetime.now()` somewhere.

**2. A config boundary.** Model ID, `max_tokens`, effort, and timeouts belong in config, not
scattered through call sites. Model IDs change every few months and you will migrate; you want
one place to edit.

**3. An eval harness.** This is the piece most people skip and the piece that distinguishes
someone who has shipped an LLM system from someone who has demoed one. Minimum viable version:
a set of realistic inputs, expected outputs or a rubric, and a script that runs the current
config across all of them and reports a score. It does not need to be sophisticated. It needs
to exist *before* you tune prompts, so that "this prompt is better" is a measurement rather
than a vibe.

Two things that only an eval harness can tell you: whether a component everyone recommends
actually helps *on your data* (my own case: cross-encoder reranking is in every RAG reference
architecture, and ablating it showed it made retrieval measurably worse on my corpus), and
whether a prompt change that improved one case regressed three others.

**4. Prompt versioning.** Prompts are code. Version them, diff them, and record which version
produced which eval score. A prompt edited in place with no history is unrecoverable.

**5. Retry and idempotency at the application layer.** The SDK retries transport failures. It
does not know that your tool already charged a credit card. For tool executions with side
effects, idempotency is yours.

**6. Rate-limit and concurrency management.** Token-bucket or semaphore around your call sites.
Note that fast mode, batch, and standard requests draw on separate limit pools, and different
model tiers have separate buckets — Opus 5 does not share Opus 4.x limits.

**7. Observability on the loop, not just the calls.** For agentic workloads the useful unit is
the *session*: how many iterations, which tools fired, how many tokens across the whole loop,
where it stopped. Per-call logs alone will not tell you that one runaway session burned a
thousand dollars.

**8. Batch API for anything not latency-sensitive.** 50% discount, up to 100k requests per
batch, results within an hour typically. If you are running an eval suite or a bulk extraction
job through the synchronous API, you are paying double for no reason.

---

## Part 12 - How to actually get this experience

Reading this gives you the vocabulary. It does not give you the thing an interviewer is
probing for, which is: have you been surprised by this API and had to fix it?

A weekend project that produces real, defensible experience — pick any dataset you already
have:

1. **Write the manual tool loop.** Two or three tools, hand-rolled `while` loop, proper
   `tool_use_id` matching and `is_error` handling. Do not use the Tool Runner yet.
2. **Add usage logging from the first call.** Every field in Part 11 item 1.
3. **Add prompt caching and verify it works** by watching `cache_read_input_tokens` go
   non-zero. Then deliberately break it — put a `datetime.now()` in the system prompt — and
   watch it go back to zero. That is the moment the prefix-match invariant stops being
   something you read and becomes something you know.
4. **Build a 20-case eval set** and score two different prompts against it.
5. **Sweep `effort` across `low`/`medium`/`high`** and put the cost-versus-quality numbers in a
   table.

That is maybe 300 lines of code, and afterwards every claim in this document is one you own.
Step 3 in particular gives you a concrete story — a measured cache hit rate, a specific
invalidator you caused and fixed — which is worth more in an interview than any amount of
correct terminology.

---

## Reference

- Messages API: `https://platform.claude.com/docs/en/api/messages`
- Prompt caching: `https://platform.claude.com/docs/en/build-with-claude/prompt-caching`
- Tool use: `https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview`
- Structured outputs: `https://platform.claude.com/docs/en/build-with-claude/structured-outputs`
- Model IDs and pricing: `https://platform.claude.com/docs/en/about-claude/models/overview`
- Errors: `https://platform.claude.com/docs/en/api/errors`

Model IDs, pricing, and parameter availability move. Query the Models API
(`client.models.list()` / `.retrieve(id)`) for live capability data rather than trusting a
cached table — including the one in Part 2 of this document.
