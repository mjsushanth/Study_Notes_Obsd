# Serving Labs 1 & 2 — LLM inference on a 24 GB Mac

Notes from `llm-serving-internals` (Notebooks 1 and 2). Measured on M5 Pro, 24 GB unified memory, MLX.

Two labs, two separate resources. They fail in different ways and you can be limited by either.

| | Lab 1 | Lab 2 |
| :-- | :-- | :-- |
| Resource | Memory **capacity** | Memory **bandwidth** |
| Limits | How many users fit at once | Tokens per second |
| Headline | 5× concurrency inversion vs parameter count | 35× prefill/decode gap |

---

# PART 1 — Memory: what bounds concurrency

## The budget equation

> `budget = weights (fixed) + KV cache (per-sequence, grows) + activations (transient)`

Three consumers, three different scaling behaviours. That difference is the whole lab:

- **Weights** — loaded once, shared by every request. Constant.
- **KV cache** — one per sequence, grows linearly with that sequence's context length.
- **Activations** — intermediate tensors inside a forward pass. Freed after, but must be resident at peak.

## Why the KV cache exists, and why it is per-sequence

Decoding token *t* requires attention over positions *1…t−1*, which needs the K and V projections of every prior position. Recomputing them each step is O(n²) redundant work across a generation, so you cache them.

The serving consequence: **weights amortise across users; the KV cache does not.** Every concurrent sequence carries its own. So concurrency is bought out of `budget − weights`, and the exchange rate is bytes per token.

## The cost model

```
KV bytes/token = 2 · L · H_kv · d_head · dtype_bytes
```

- **2** — K and V are separate tensors, both retained.
- **L** — every layer runs its own attention with its own projections, so each caches independently.
- **H_kv** — the number of **key/value** heads. Not query heads. This is the factor people get wrong.
- **d_head** — dimension per head.
- **dtype_bytes** — 2 for fp16/bf16.

Sanity check on a toy: `L=2, H_kv=2, d_head=4, 2 bytes` → `2·2·2·4·2 = 64 bytes/token`. Cheap to do, catches factor-ordering mistakes before a real model hides them.

## Query heads vs KV heads (why GQA is asymmetric)

The `H_kv` factor above only makes sense once the asymmetry is clear.

**Multi-head attention (MHA).** For each of `n_heads` heads you project the input into a query, a key, and a value: `Q_h, K_h, V_h`. Then `head_h = softmax(Q_h K_hᵀ / √d) V_h`. Symmetric — one of each per head.

**The asymmetry is temporal, not structural:**

| | Lifetime | Cached? |
| :-- | :-- | :-- |
| `Q` | computed for the current token, used in this step, discarded | no |
| `K`, `V` | must be available to **every future step** of this sequence | yes |

Query is transient state. Key and value are persistent state. So the memory cost of attention sits entirely on the K/V side, and cutting query heads buys you compute but no memory.

**Grouped-query attention (GQA).** Keep `n_heads` query projections, but only `n_kv_heads` key/value projections. A group of `n_heads / n_kv_heads` query heads shares one K/V pair.

Llama-3.1-8B: 32 query heads, 8 KV heads → group size 4 → **4× smaller cache**. 512 KB/token becomes 128 KB/token.

Why this trade is cheap in quality: attention *patterns* are what differentiate heads, and the pattern comes from `Q_h K^T` — so 32 query heads still produce 32 distinct attention distributions even over a shared key space. What is lost is the ability to project keys and values into 32 independent subspaces. Empirically that costs little. Pushing to `n_kv_heads = 1` (multi-query attention, MQA) does degrade noticeably; GQA is the interior point that kept most of the memory win.

> [!note] Every model in the table below has `H_kv = 8`.
> GQA with 8 KV heads has effectively become the industry default. So all the variation in cache cost across current models comes from **`L` and `d_head` only** — which is why it decouples from parameter count so cleanly.

## Parameter count does not predict cache size

Two different products:

```
params    ~  L · d_model²
KV/token  ~  L · H_kv · d_head
```

They share `L` and nothing else. `d_model²` and `H_kv · d_head` move independently across architectures, so there is no reason to expect model size to predict serving capacity — and it doesn't.

| Model | Weights | L | H_kv | d_head | KV KB/token |
| :-- | --: | --: | --: | --: | --: |
| gpt-oss-20b | 12.08 GB | 24 | 8 | 64 | **48** |
| Qwen3-14B | 8.31 GB | 40 | 8 | 128 | 160 |
| DeepSeek-R1-14B | 8.31 GB | 48 | 8 | 128 | 192 |
| gemma-3-12b | 8.03 GB | 48 | 8 | 256 | **384** |
| Llama-3.1-8B | 4.52 GB | 32 | 8 | 128 | 128 |

Since `H_kv = 8` and `dtype = 2` throughout, the formula collapses to a one-liner for this table:

```
KV KB/token = (L · d_head) / 32
```

- gpt-oss-20b: `24 · 64 / 32 = 48`
- gemma-3-12b: `48 · 256 / 32 = 384` — 2× the layers, 4× the head dim, **8×** the cost

Converting to concurrency:

```
max_sequences = (budget − weights) / (bytes_per_token · context_len)
```

**At 4k context, gpt-oss-20b serves ~5× more concurrent sequences than gemma-3-12b, while being 50% larger in parameters.** That inversion is the load-bearing result of Lab 1.

**Practical rule:** "pick the smaller model for throughput" is not valid. Read `config.json` for `L` and `head_dim` first.

## Verification

Allocate a cache of known token count, read the allocator delta:

| Model | Derived | Measured |
| :-- | --: | --: |
| gpt-oss-20b | 48.000 KB | 48.005 KB |
| gemma-3-12b | 384.000 KB | 384.004 KB |

The formula holds. That same measurement settles `dtype_bytes` empirically: **2 bytes, on models whose weights are 4-bit quantised.**

## Quantisation does not shrink the KV cache

Expected once stated precisely: quantisation is applied to **weight tensors**. K and V are activations, produced at runtime by the projection layers, and they materialise at the compute dtype.

Consequence worth remembering:

> Quantisation shrinks the weights term and leaves the KV term untouched, so **it shifts the bottleneck toward KV.** A 4-bit model has proportionally *less* headroom per byte of weights than its unquantised equivalent.

Quantising weights buys you a bigger model, not more concurrent users. Getting more users needs a separate mechanism — quantised KV, which this lab does not cover.

## Where the formula breaks: activations

Gate S1.3 compared the derived ceiling against an empirical one (allocate caches until allocation fails). It failed, and was left failed:

| Model | Derived | Empirical | Budget at run time |
| :-- | --: | --: | --: |
| gpt-oss-20b | 29 | **15** | 17.67 GiB |
| gemma-3-12b | 4 | **3** | 14.04 GiB |

Roughly 2× off on the large model. The missing term is activations, and the magnitude is the surprise:

> A 4096-token prefill cost **~1.66 GB of activations against ~192 MB of KV** — about **8.6×** the cache it produced.

The reason is a scaling difference:

- **KV cache** scales with tokens **stored**.
- **Activations** scale with tokens processed **simultaneously in one pass**, plus `d_model` and the MLP expansion factor.

A 4096-token prefill processes all 4096 positions at once, so the activation term is driven by **chunk size** — a *scheduling* parameter. This is the seam between the two labs: `max_num_batched_tokens` has a memory cost, not just a latency one.

**Correct reading:** the KV-only formula is an **upper bound**, not a prediction. It still tells you which architecture wins. It does not tell you the achievable ceiling, which sits below it by an amount your chunking policy sets.

---

# PART 2 — Time: what bounds throughput

## Prefill and decode: same weights, two shapes

Both phases run the identical model. The difference is the shape of the input, and it changes everything.

| | Prefill | Decode |
| :-- | :-- | :-- |
| Processes | all N prompt tokens, one pass | 1 token per sequence per step |
| Core op | **GEMM** — `(N, d) × (d, d)` | **GEMV** — `(1, d) × (d, d)` |
| Each weight element is | loaded once, reused across N rows | loaded once, used **once** |

- **GEMM** — GEneral Matrix–Matrix multiply, `C = AB`.
- **GEMV** — GEneral Matrix–Vector multiply, `y = Ax`.

## Why decode is slow

Decode at batch size 1: you stream the **entire** weight set out of memory — call it `W` bytes — to perform ~`2W` FLOPs. That is roughly **1 FLOP per byte loaded**.

The hardware can do hundreds of FLOPs in the time it takes to fetch one byte. So the step duration is set by *how fast weights arrive*, not by arithmetic. The compute units are idle most of the step. This is what **memory-bandwidth-bound** means. (The formal name for FLOPs-per-byte is *arithmetic intensity* — worth recognising, not worth building on here.)

Prefill amortises the same weight traffic across N tokens, so it is not in this regime.

Measured, Llama-3.1-8B:

| | Throughput |
| :-- | --: |
| Prefill | 1,925 tok/s |
| Decode | 55 tok/s |
| **Ratio** | **35×** |

## Batching is the fix — and this is why

If the weight read is the cost, and it is being spent to produce a single token, then the obvious move is to produce more tokens per read.

Batch `B` sequences: the GEMV becomes a GEMM with `B` rows. **Same weight traffic, B times the output.** Returns are near-linear until you have enough arithmetic to saturate the compute units, then they flatten.

> Batching is not a throughput trick layered on top of inference. It is the direct remedy for the 35× gap.

## Static vs continuous batching

A **slot** is one position in the batch — capacity for one in-flight sequence.

**Static (request-level) batching.** Form a batch, run until *all* members finish, then form the next. A slot vacated by a short sequence sits idle until the longest member of that batch completes.

**Continuous (iteration-level) batching.** Re-form the batch every forward pass. Evict finished sequences, admit waiting ones immediately.

The cost of static batching only appears when output lengths **vary**. Under uniform lengths every slot drains together and there is nothing to recover — which is why the lab generates output lengths from a **lognormal** distribution. Real traffic is heavy-tailed: mostly short completions, a long thin tail. Using uniform lengths here would make the entire result vanish.

Same workload, same seed, scheduler as the only variable:

| Scheduler | Idle slot fraction |
| :-- | --: |
| Static | 0.491 |
| Continuous | **0.125** |
| | **74.6% relative reduction** |

Two checks make this believable rather than merely favourable:

1. **Total output tokens are identical.** Otherwise the "faster" scheduler might just be doing less work.
2. **21 admissions occurred after step 0.** A continuous scheduler that never admits mid-flight is functionally static — it would show a plausible improvement from nothing. Counting mid-flight admissions is the direct test that the mechanism actually engaged.

## The three admission constraints

A waiting request is admitted only if it clears all three simultaneously:

1. `max_num_seqs` — concurrent sequence cap (slots)
2. `token_budget` — total tokens, decode **and** prefill, schedulable per iteration
3. **KV capacity** — from Part 1

The scheduler records which one blocked it each iteration and exposes it as `step_binding_constraint`.

Worth dwelling on, because production tooling generally does not surface this. When throughput degrades, the default logs will not tell you whether you are slot-limited, budget-limited, or cache-limited — and **the fix is completely different in each case.** Making it an observable turns a guess into a measurement.

## Head-of-line blocking

Inject one 8,192-token prompt into a stream of short requests, chunked prefill disabled.

Without chunking, a request is admitted only if its **entire** prompt fits the remaining per-iteration token budget. 8,192 tokens never fits. And because admission is FIFO with no skip-ahead, nothing behind it is considered either.

Measured admission delay: **unbounded**. Not a spike — permanent starvation of the queue for the rest of the run.

Two independent causes, and fixing *either* resolves it:

- the request is unschedulable as a unit → **chunked prefill**
- the queue discipline will not bypass it → **skip-ahead / priority**

## Chunked prefill: one knob, two metrics

Split the prefill across iterations — process `max_chunk` tokens per step against a persistent cache. The long request progresses incrementally; the queue drains.

Sweeping `max_chunk`:

| max_chunk | Long-request TTFT (iters) | P99 ITL (ms) |
| --: | --: | --: |
| 8 | 1,023 | 76 |
| 128 | 63 | 135 |
| 2048 | **3** | **1,132** |

TTFT improves ~341× across the full sweep; P99 inter-token latency degrades ~15×. **One parameter, two metrics moving in opposition.**

There is no optimum without a workload target:

- **Interactive serving** is ITL-sensitive. A one-second stall mid-generation reads as a failure. Small chunks.
- **Batch inference** cares about completion time and is largely indifferent to ITL. Large chunks.

In vLLM this parameter is `max_num_batched_tokens`. It also moves peak activation memory (Part 1), so it is a memory knob and a latency knob at the same time.

## Footnote: the scheduler orders decode before prefill

Every running sequence gets its one decode token *before* any admission is considered. A deliberate starvation guarantee — an arriving long prompt cannot stall existing users.

Small trap that follows from it. Measured **in iterations**, ITL is exactly 1.00 token/step regardless of what prefill is doing — the ordering guarantees it. But iteration *duration* is not constant: an iteration scheduling 2 decode tokens plus 510 prefill tokens takes far longer than one scheduling 2 decode tokens.

> The scheduler is fair **per iteration** and unfair **per second**. Per-iteration accounting cannot observe prefill/decode interference, because it discards duration before the question is asked.

In wall-clock across the sweep: **mean ITL degrades 1.9×, P99 degrades 15×.** The interference concentrates in the tail — which is where latency SLOs are written. Reporting the mean alone understates it by ~8×.

---

# Recall checklist

Blank sheet, no lookups:

- [ ] Write the KV bytes/token formula. Explain why `n_kv_heads` and not `n_heads` — in terms of what gets cached and what gets discarded.
- [ ] Explain why quantising weights does not shrink the KV cache, and which way that moves the bottleneck.
- [ ] Given `L` and `d_head` from a config you have not seen, predict the concurrency ceiling — and state why your answer is an upper bound.
- [ ] Explain why decode is memory-bandwidth-bound and prefill is not, and why that one fact is the entire argument for batching.
- [ ] Sketch static vs continuous batching. Name the workload property that makes the difference appear at all.
- [ ] Name the three admission constraints, and say what you would change if each one were binding.
- [ ] State the chunked-prefill tradeoff in TTFT/ITL terms, and pick a side given a product.
