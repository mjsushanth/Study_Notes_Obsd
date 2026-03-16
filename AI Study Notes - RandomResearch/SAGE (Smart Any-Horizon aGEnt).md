

- AllenAI’s answer to a very specific failure mode in video-LLMs: if you force a model to “watch” a long video by uniformly sampling more and more frames and answering in one shot, accuracy eventually saturates or drops while compute explodes. SAGE flips the paradigm: treat long-video reasoning as an **agentic process** that can either answer immediately (when the question is short-horizon) or **decide to take multiple tool-augmented steps** (when the question is long-horizon), much closer to how humans skim/rewind/search.

```

(128 frames + metadata + query)
          |
          v
   [Stage-1: gate]
    |          |
 answer now   call tool
    |          |
    v          v
  response   [Stage-2 loop up to 10]
                 |
      tool(action JSON) -> runtime executes -> observation
                 |
                 v
          update context, decide next tool or stop

```


## What “any-horizon” means (the core idea)

A long video question has an implicit “information horizon”: sometimes the evidence is in the first few seconds; sometimes it’s buried at minute 18; sometimes you need speech; sometimes you need outside knowledge (e.g., a cast list, match result). The “any-horizon” property is: **the system adapts its reasoning depth and evidence gathering to the query**, rather than always paying the worst-case cost.

Concretely, SAGE is designed to choose between:

- **Single-turn**: answer from a small initial context (sampled frames + metadata) when that’s enough.
- **Multi-turn**: iteratively call tools to fetch the right segment(s), transcript, or external facts, then answer.

This “mixture of behaviors” is not just prompting; they train the orchestrator so it learns _when_ to stop and _when_ to keep searching.

## System architecture: SAGE + an orchestrator (SAGE-MM)

The system takes four inputs: **128 sampled frames**, **video metadata** (path/duration), **tool definitions**, and the **user query**. Then it runs a two-stage controller called **SAGE-MM** (“MM” as the multimodal orchestrator).

### Stage 1: “Context VLM” decision (one-step gate)

SAGE-MM produces a _structured JSON action_ that includes:

- `video-context` (what’s going on in the video at a high level),
- `query-intent`,
- `recommended-tool` (what to do next if not answerable now),
- `final-answer` (or null if it wants tools).

This is the **any-horizon gate**: it explicitly decides “I can answer now” vs “I need to act.”

### Stage 2: iterative tool loop (up to 10 steps)

If tools are needed, SAGE-MM enters a loop. At each step it outputs JSON with:

- `answerable` (boolean),
- `recommended-tool`,
- `final-answer` (or null).  
    They cap this loop at **10 steps** to avoid infinite runs.

---
