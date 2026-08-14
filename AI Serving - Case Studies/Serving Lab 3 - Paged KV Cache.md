# Serving Lab 3 — Paged KV cache

Notes from `llm-serving-internals`, Notebook 3 (`serving3_paged_kv.ipynb`, 26 cells). Measured against `src/serving/blocks.py`; results in `artifacts/nb3_llama-3.1-8b.json`.

This notebook needs no model at all. `serving.blocks` is pure logic — a pool of integers and a per-sequence list of integers — so every number below is reproducible on any machine in milliseconds. That is worth noticing: the most consequential memory result in the whole lab does not require a GPU to establish.

The three labs ask three different questions about the same fixed budget:

| | Lab 1 | Lab 2 | Lab 3 |
| :-- | :-- | :-- | :-- |
| Question | How many sequences **fit**? | How fast do tokens **come out**? | Of what you reserved, how much **holds a token**? |
| Resource | memory capacity | memory bandwidth | capacity **management** |
| Headline | 5× concurrency inversion vs parameter count | 35× prefill/decode gap | **0.108** utilisation under naive reservation |

Lab 1 computed a ceiling. Lab 2 found that the scheduler can be blocked by that ceiling. Lab 3 asks whether the ceiling was ever real — and finds that under the obvious allocation policy, roughly nine tenths of the reserved cache is holding nothing.

---

# PART 1 — The waste

## The workload every number comes from

One workload generates every measurement in this notebook, so it is worth pinning down first.

```
N_REQUESTS   = 40
SEED         = 0
dist         = lognormal
prompt_mean  = 8
output_mean  = 64
```

Measured output lengths: **min 5, mean 55.325, max 453, total 2213 tokens.**

Two things about that shape matter later.

**Lognormal, not uniform.** A lognormal distribution is one whose logarithm is normally distributed — concretely, mostly small values with a long thin tail of large ones. The sorted lengths make it visible: twelve replies under 16 tokens, then one of 453. Real chat traffic looks like this. If output lengths were uniform, several results in this notebook would shrink or vanish, exactly as continuous batching's win vanished under uniform lengths in Lab 2.

**Total tokens = 2213.** Memorise this one number. Every utilisation figure in the notebook is `2213 / (something)`, and once you see that, the whole block-size sweep becomes a single division you can do yourself.

## Why the naive scheme is the natural thing to write

The naive policy is: **when a request arrives, reserve one contiguous span of KV capacity large enough for the longest reply it could produce.** This is not laziness. Two independent constraints push you there.

**Constraint 1 — attention kernels want contiguity.** The attention step for one sequence computes `softmax(q · Kᵀ) V` over every cached position. `K` and `V` are tensors, and a tensor in any array library is a base pointer plus a stride pattern. A GPU kernel reading a contiguous span issues coalesced loads: consecutive threads read consecutive addresses, which the memory subsystem services as a small number of wide transactions. Scatter those positions across the address space and you need either an indirection table consulted inside the kernel's inner loop, or a separate gather pass that copies the pieces into a contiguous scratch buffer before the kernel runs. Both cost something, and neither is what `mlx_lm`'s or PyTorch's stock attention does. Contiguous is the shape the fast path already accepts.

**Constraint 2 — you do not know the output length.** You must commit the reservation *before* generation starts, and the reply's length is decided by the model, one token at a time, and is only known when it emits an end-of-sequence token. There is no oracle. So the only length you can safely reserve is an upper bound you choose in advance.

Put those together and max-length reservation is the *correct* engineering response to the constraints as stated. The interesting move is not to write a smarter length predictor. It is to attack Constraint 1 — to stop needing contiguity at all.

> Max-length reservation is what you get when you accept that attention wants contiguous memory and that output length is unknowable. Paging refuses the first premise rather than trying to fix the second.

## The measurement: 0.108

The notebook picks a ceiling and divides.

```
MAX_LENGTH_CEILING = 512

utilisation_i    = output_len_i / 512
mean_utilisation = mean over the 40 requests
                 = 2213 / (40 × 512)
                 = 2213 / 20480
                 = 0.108056640625
```

**Gate S3.1** required mean utilisation under 0.35. Measured **0.108** — it passed by a factor of three.

Read that as: **under 11% of the reserved KV cache holds a real token.** Equivalently, you provisioned a machine for 40 concurrent users and are getting the memory efficiency of about 4.3.

> [!note] The 512 ceiling is a *generous* choice, and that makes the result stronger.
> Llama-3.1's actual `max_position_embeddings` is 128k. The notebook does not reserve that — it reserves 512, described in the code as "a plausible reserve-for-the-practical-max-reply policy." So 0.108 is what you measure after already being sensible. A literal `max_seq_len` reservation would put utilisation in the 10⁻³ range. When a number is embarrassing, check whether the setup was charitable before you trust it; here it was, which is why the number counts.

Why does intuition get this wrong? Because intuition averages the *ratio* by imagining a typical case, and the arithmetic averages the *lengths*. Mean length is 55.325 against a 512 ceiling. The single 453-token reply — the one your intuition anchors on, because it is the one that would have justified the reservation — is 1 request in 40. The other 39 pay its bill.

## The oracle bound, and why it is the honest comparison

An **oracle** here means a policy with perfect foreknowledge of each request's true final output length. Given that, it reserves exactly `output_len` and not one slot more.

```
oracle_utilisation = 1.0    # by construction: reserved == used, always
```

This is not measured. It is 1.0 by definition, hardcoded in the notebook as such. So what is the point of it?

**It fixes the denominator of the claim.** Without a ceiling, "0.108 is bad" is an assertion about your own taste. With it, you can say precisely: *0.108 is 10.8% of what is achievable in principle*, and — this is the part that matters — *the 0.892 gap is attributable entirely to not knowing the future, not to any other inefficiency*. An oracle bound separates "this policy is wasteful" from "this problem is hard." Here the problem is not hard in an information-theoretic sense; a policy with the right data structure recovers most of the gap without predicting anything.

This is a general move worth stealing. Whenever you report an efficiency, ask what the unattainable-but-well-defined best case is. If you cannot state it, you do not yet know what you are measuring against.

## Two kinds of fragmentation

**Fragmentation** is unusable memory: capacity you hold but cannot put a token into. There are two distinct kinds, they have different causes, and fixing one does not fix the other. Getting this distinction crisp is the load-bearing conceptual step of the notebook.

| | Internal fragmentation | External fragmentation |
| :-- | :-- | :-- |
| Where the waste sits | **inside** a span you were granted | **between** spans, in the free pool |
| Cause | you asked for more than you needed | free space exists but not in the *shape* required |
| Naive scheme's version | 512 reserved, 55 used → 457 slots idle | 224 tokens free in total, largest single gap 51, a 52-token request refused |
| Fixed by right-sizing? | yes — this is exactly what right-sizing fixes | **no** — variable sizes are the cause |
| Bound under naive alloc | unbounded: scales with the ceiling | unbounded: depends on free/alloc history |

**Internal fragmentation, precisely.** Space inside an allocation that holds no token. Under max-length reservation it is `ceiling − used` per sequence, which scales with the ceiling you chose. Choose a bigger ceiling for safety and you buy proportionally more waste. That is what 0.108 measured.

**External fragmentation, precisely.** A request fails even though total free capacity exceeds what it asked for, because no single free region is large enough. The failure is about the *geometry* of the free space, not its quantity. The classic parallel is `malloc` on a long-running process: allocate and free variable-size objects in unpredictable order for long enough, and the heap becomes a lace of gaps that are individually too small for anything useful.

The crucial asymmetry: **you can fix internal fragmentation completely by right-sizing every reservation, and external fragmentation will still be there** — worse, in fact, because right-sizing means every reservation is a different size, which is precisely what shreds the free space.

## External fragmentation, measured — and the gate clause that saved it

The notebook writes a toy contiguous allocator: a free list of `(start, length)` gaps, first-fit placement, requests released in random order (40% chance per admitted request that a randomly chosen live one finishes). Then it counts admission failures.

Two design decisions here are worth more than the result.

**Decision 1: how to size the pool.** The first version of this cell sized the pool to *total* demand — the sum of all 40 output lengths, 2213. It measured nothing: zero failures. The bug is that total demand is not what a pool must cover, because requests finish and return their space. **Peak concurrent demand** is what matters, and it is far smaller. So the notebook now measures it first, in a dry run against an effectively unlimited pool:

```
peak_live_demand = 1558 tokens      # measured, dry run, pool_size = 10**9
POOL_SIZE        = int(1558 × 0.8) = 1246
```

The pool is then deliberately sized *below* peak demand, so the run is genuinely contended rather than comfortably oversized.

**Decision 2: what counts as a failure.** This is the good part. Running out of memory is not external fragmentation — it is a pool that is too small, which is a boring and different claim. So the gate has two clauses:

```
gate = (len(failure_snapshots) > 0)                    AND
       (total_free_at_failure > requested_length)      for at least one failure
```

The second clause *is* the definition of fragmentation: enough space, wrong shape. Measured:

| | Value |
| :-- | --: |
| `external_fragmentation_failures` | **2** |
| `..._that_are_true_fragmentation` | **1** |

So one of the two failures was **scarcity**, not fragmentation: at that moment total free space was also less than the request needed, and no allocator on earth could have served it. The other was genuine: from the ledger's record of the run, **224 tokens free in total, largest single gap 51, a 52-token request refused.** One token short in the biggest gap, with more than four times the needed space sitting in the pool.

Now the lesson. A one-clause gate — `failures > 0` — would have passed on this run and reported "external fragmentation demonstrated." That report would have been half wrong, and nothing in the output would have shown it, because a failure looks identical either way from the outside. The two-clause gate is a measurement that **can fail correctly**: it distinguishes the phenomenon you claim from the mundane thing that resembles it.

> A measurement that cannot come back negative is not a measurement. Before running one, write down what the *null* result looks like — here, "a failure that was merely scarcity" — and make sure your instrument can tell it apart from the finding.

The notebook also names a way its own setup is unfair: `release()` never **coalesces** adjacent free gaps back into a single larger one, whereas real allocators do. That overstates fragmentation. It is a legitimate worst-case demonstration, but only because it is admitted as one rather than left for a reader to discover.

## What 1246 against 1558 actually means

`POOL_SIZE = 1246` and `peak_live_demand = 1558` are in the same units — tokens of contiguous capacity in the Step 3 toy allocator. Do not confuse them with the paged pool later on, which is measured in blocks (`n_blocks=200` at `block_size=16`).

Demand exceeded supply by 312 tokens, deliberately. So what does a real server do at that moment?

It cannot simply refuse the request, because the request may already be half generated. The real answer is **preemption**: pick a victim sequence, take its KV back, and reinstate it later. Two ways to reinstate:

- **Swap** — copy the victim's KV blocks out to host memory, free the device blocks, copy back on resume. Costs bandwidth in both directions, preserves the exact tensors.
- **Recompute** — drop the victim's KV entirely, and on resume re-prefill its prompt plus the tokens it had already generated. Costs a prefill, which from Lab 2 is the cheap-per-token phase, and needs no extra host memory.

vLLM implements both and picks per configuration. The tradeoff is roughly bandwidth against compute, and recompute usually wins for short sequences because prefill is throughput-efficient.

**Be clear about what the notebook does here: nothing.** `BlockAllocator.allocate()` raises `OutOfBlocks` when the free list is empty, and the docstring says exactly what that stands in for — *"In a real server this triggers preemption."* There is no victim selection, no swap, no recompute, no eviction policy anywhere in `blocks.py`. Exhaustion is signalled and then handed to a caller that does not exist. That is a real boundary of the lab, not an oversight to gloss over.

---

# PART 2 — The mechanism

## Blocks and the block table

The fix is one structural change: **stop allocating contiguous variable-size spans; allocate fixed-size blocks, and give each sequence a table mapping its logical positions onto whichever physical blocks it happens to hold.**

Three definitions, one sentence each:

- A **block** is a fixed-size chunk of KV capacity, holding `block_size` token-slots. Every block in the pool is the same size, which is what makes them interchangeable.
- The **physical block id** is an index into the global pool — which piece of memory this is.
- The **logical block index** is a per-sequence counter, 0, 1, 2, … — which piece of *this sequence's* history it holds.

A sequence's **block table** is just the list of physical ids in logical order. That is the whole idea:

```python
class BlockTable:
    """Per-sequence logical -> physical block map. The whole idea of paging, in one dict."""
    self._blocks: list[int] = []   # logical index -> physical block id
    self._n_tokens = 0
```

## The indexing arithmetic

Given a token position `pos` (0-based within the sequence), the address of its KV slot is:

```
logical_idx, offset = divmod(pos, block_size)
physical_block      = block_table[logical_idx]
slot                = (physical_block, offset)
```

Written out: `logical_idx = pos // block_size` says *which* of my blocks, and `offset = pos % block_size` says *where inside it*. The division is the only work. In `blocks.py`:

```python
def slot_for_position(self, pos: int) -> tuple[int, int]:
    """Map a token position to (physical_block, offset within that block)."""
    if not 0 <= pos < self._n_tokens:
        raise IndexError(f"position {pos} outside 0..{self._n_tokens - 1}")
    logical_idx, offset = divmod(pos, self._block_size)
    return self._blocks[logical_idx], offset
```

Note the two-level structure. `pos → (logical_idx, offset)` is pure arithmetic, identical for every sequence. `logical_idx → physical_block` is a table lookup, different for every sequence. All the per-sequence state lives in that one list.

## The line that is the entire mechanism

```python
def append_token(self) -> None:
    """Allocates a new physical block only when the current tail block is full."""
    if self._n_tokens % self._block_size == 0:
        self._blocks.append(self._allocator.allocate())
    self._n_tokens += 1
```

`self._n_tokens % self._block_size == 0` is true exactly when the current tail block is exactly full (and when the sequence is empty). At that instant, and only then, claim one more block.

**Nothing is reserved ahead of need, ever.** A sequence holds precisely as much capacity as it has produced, rounded up to a block boundary. A reply that turns out to be 12 tokens long never paid for the 512 it might have been. Everything downstream in this notebook is a consequence of that one conditional — it is not a collection of separate tricks.

## Worked example, by hand

`block_size = 4`, a fresh 10-block pool, three requests grown to 6, 3, and 9 tokens in that order. The allocator hands out the lowest free id first, deliberately, so this is checkable on paper:

```text
request A (6 tokens):  logical 0 -> physical 0    logical 1 -> physical 1
                       2 blocks = 8 slots, 6 held, 2 wasted
request B (3 tokens):  logical 0 -> physical 2
                       1 block  = 4 slots, 3 held, 1 wasted
request C (9 tokens):  logical 0 -> physical 3    logical 1 -> physical 4    logical 2 -> physical 5
                       3 blocks = 12 slots, 9 held, 3 wasted
```

Trace A's addresses: position 0 → `divmod(0,4)` = `(0,0)` → block 0, offset 0. Position 3 → `(0,3)` → block 0, offset 3. Position 4 → `(1,0)` → block **1**, offset 0. Position 5 → `(1,1)`. The block boundary crossing between positions 3 and 4 is invisible to the sequence and is a jump to a different physical block.

And the point that motivates the whole design: **A holds physical blocks 0 and 1, which happen to be adjacent. They did not have to be.** In `test_blocks.py`:

```python
def test_blocks_are_not_contiguous_when_the_pool_is_shared(self):
    grow(first, 1)   # takes block 0
    grow(second, 1)  # takes block 1
    grow(first, 4)   # tokens 2-4 fit block 0; token 5 opens block 2
    assert first.blocks == (0, 2)
    assert second.blocks == (1,)
```

Two sequences interleaving their growth get interleaved physical blocks. Sequence one owns `(0, 2)` — a hole in the middle of its own memory, and it does not care.

## What paging does to each fragmentation, exactly

**Internal fragmentation: unbounded → bounded.** From the `wasted_slots` property:

```python
@property
def wasted_slots(self) -> int:
    """Capacity of allocated blocks minus tokens held."""
    return len(self._blocks) * self._block_size - self._n_tokens
```

Since `len(self._blocks) == ceil(n_tokens / block_size)`, the waste is `(-n_tokens) mod block_size`, which is at most `block_size − 1`. Compare the two regimes:

```
naive:  waste per sequence  =  ceiling − used          (scales with the CEILING you chose)
paged:  waste per sequence  ≤  block_size − 1          (scales with NOTHING; it is a constant)
```

That change of what the waste scales with is the entire argument. Total waste stops being a function of your safety margin and becomes a function of how many sequences you are running, times a small constant you control directly. At `block_size = 16` the worst case is 15 slots per sequence — against 457 for the mean-length sequence under the 512 ceiling.

**External fragmentation: eliminated.** Not reduced — structurally gone. External fragmentation requires that free space have a *shape* that can mismatch a request. When every block is the same size and any block can serve any logical index of any sequence, there is exactly one shape. A request for `k` blocks succeeds if and only if `k` blocks are free anywhere. The free list can never be "large enough in total but wrong in arrangement," because arrangement no longer exists as a property.

**What paging does not do: it does not create capacity.** You can still exhaust the pool — that is `OutOfBlocks`, and it is scarcity, the honest failure. Paging converts a confusing failure mode into a clear one. The clear one still needs preemption to handle.

## The virtual-memory parallel, and where it stops

The parallel is exact enough to be worth using: this is the OS's page table, and `blocks.py` is a page table. Same problem (contiguous-looking address space over non-contiguous physical memory), same solution (fixed-size pages plus a per-process translation table), same win (external fragmentation eliminated, internal fragmentation bounded by page size). `fork()` with copy-on-write shows up later in this note for the same reason.

Where the analogy breaks, and it matters for reasoning about cost:

- **No hardware translation.** A CPU has a memory management unit that walks page tables in silicon, and a TLB (translation lookaside buffer, a small cache of recent translations) that makes the common case nearly free. Here the block table is a Python list walked in software, and in a real implementation it is a tensor of block indices consulted by a kernel. **Translation is not free, and its cost is the reason `block_size` cannot go to 1 in practice.**
- **No page faults, no backing store.** OS paging can evict a page to disk and fault it back in transparently on the next access. There is no equivalent here: a block is either in the pool or the allocation fails loudly with `OutOfBlocks`. The nearest analogue is vLLM's swap-to-host-memory, and it is explicitly scheduled by the server, not triggered by a trap on access.
- **No protection or sharing semantics from the kernel.** Refcounting and copy-on-write are implemented by hand in this module, in Python, and are only as correct as that code — which is exactly how defect F2 below happened.

## The measured operating point

Block size 16, a 200-block pool, and — importantly — **overlapping sequence lifetimes**: a fixed window of 4 concurrently-live sequences, each released back to the pool the moment it finishes.

| Metric | Value | What it counts |
| :-- | --: | :-- |
| `block_utilisation` | **0.8754** | tokens held ÷ capacity of blocks allocated, summed over all 40 sequences |
| `total_blocks_ever` | **158** | logical blocks requested, summed over all 40 sequences, over the whole run |
| `peak_allocated` | **29** | high-water mark of blocks allocated *simultaneously* |

These three are easy to blur together, so pin down each one.

`total_blocks_ever = 158` is a **cumulative** count: `Σ ceil(len_i / 16)` over the 40 requests. It counts *allocation events* across time. It says nothing about how much memory was ever resident at once.

`peak_allocated = 29` is an **instantaneous maximum**: the largest number of blocks in the pool that were simultaneously in use. This is the number that sizes your hardware. Note it is 29, well under the 200-block pool — the pool was comfortable.

`block_utilisation = 0.8754` is derivable from `total_blocks_ever` and nothing else:

```
utilisation = total_tokens / (total_blocks_ever × block_size)
            = 2213 / (158 × 16)
            = 2213 / 2528
            = 0.8753955696202531        ← matches the artifact exactly
```

**Gate S3.2** required two things: utilisation above 0.85, **and** `peak_allocated < total_blocks_ever`. The second clause is another null-result guard, and a good one. If blocks were handed out and never returned, then every allocation would still be live at the end and `peak_allocated` would equal `total_blocks_ever`. Measured `29 < 158` proves reuse actually happened — the same physical blocks served many sequences over the run. The same guard is asserted directly in the tests, with the failure mode named in the docstring:

```python
def test_blocks_are_genuinely_reused_across_sequences(self):
    """The null signature for gate S3.2: peak == total requested means no reuse."""
    ...
    assert pool.peak_allocated == 2   # two blocks served all five sequences
    assert pool.peak_allocated < total_requested   # 2 < 10
```

This is also why the notebook uses a concurrency window instead of running the 40 sequences one at a time. One at a time, reuse is trivially perfect and the measurement is uninteresting; the window makes the pool genuinely shared.

**0.108 → 0.875.** Same workload, same seed, same total tokens. The only thing that changed is the data structure. That is the result of the notebook.

## The block-size sweep

Sweep `block_size` over powers of two and watch two costs move in the same direction — which is what makes it a real tradeoff rather than a free lunch.

| `block_size` | Utilisation | Block-table entries | Waste bound / seq |
| --: | --: | --: | --: |
| 1 | **1.0000** | 2213 | 0 |
| 4 | 0.9706 | 570 | 3 |
| 8 | 0.9377 | 295 | 7 |
| **16** | **0.8754** | **158** | 15 |
| 32 | 0.7517 | 92 | 31 |
| 64 | 0.5962 | 58 | 63 |
| 128 | 0.3842 | 45 | 127 |

Every entry in the utilisation column is `2213 / (entries × block_size)`. Verified against the artifact to the last digit for all seven points. If you remember 2213, you can rebuild this table.

**Why `block_size = 1` gives exactly 1.0 — derive it, do not memorise it.** With `block_size = 1`, each block holds one token-slot. So `ceil(n / 1) = n`: a sequence of `n` tokens holds exactly `n` blocks, whose total capacity is `n × 1 = n` slots, all of them occupied. Waste is `block_size − 1 = 0` by the bound, identically, for every sequence. Utilisation is `2213 / (2213 × 1) = 1`. Note what this means: **`block_size = 1` reaches the oracle bound of 1.0 without any foreknowledge whatsoever.** Perfect memory efficiency was never blocked by not knowing the future. It was blocked by insisting on contiguity.

**Why nobody sets it to 1.** Look at the entries column: 2213 against 158, a 14× increase in bookkeeping. That cost is not merely storage. Every entry is:

- **Metadata memory.** The block table itself is state, per sequence, and on a GPU it is a tensor that must be resident and kept up to date.
- **Transfer per step.** In a real implementation the block tables of all running sequences are handed to the attention kernel each forward pass. Longer tables mean more of them to build, upload, and index.
- **Indirection inside attention.** Every block boundary is a pointer chase in the kernel's inner loop. At `block_size = 1` there is a lookup per token — you have replaced coalesced contiguous reads with fully scattered ones, which is exactly the property that made contiguity attractive in the first place. At `block_size = 16`, one lookup amortises over 16 consecutive slots that *are* contiguous, so the inner loop still gets wide reads.

That last point is the real one, and it is why the analogy to OS pages holds: page sizes are 4 KB, not 4 bytes, for the same reason.

**Why 16 in particular.** vLLM's default `block_size` is 16. Read it off the curve: utilisation from 1 to 16 costs 12.5 percentage points and buys a 14× reduction in table length. From 16 to 32 you save only a further 1.7× in table length and pay 12.4 more points of utilisation. The curve's knee is right there — utilisation is still high, indirection is already amortised 16-wide, and it is a power of two so the `divmod` is a shift and a mask.

> [!note] The sweep's numbers depend on *your* workload, and the mechanism of that dependence is worth understanding.
> Waste per sequence is `(−n) mod block_size`. If lengths were spread evenly relative to `block_size`, mean waste would be about `(block_size − 1)/2` and utilisation would be roughly `mean_len / (mean_len + (block_size−1)/2)`. At `block_size = 16` that heuristic gives 0.881 against the measured 0.875 — close. At `block_size = 128` it gives 0.466 against the measured 0.384 — badly off, because the assumption has broken: mean length is 55, so 37 of the 40 sequences fit inside a *single* 128-slot block and waste `128 − n` rather than a uniform-ish remainder. **What actually governs utilisation is the ratio of `block_size` to typical sequence length.** A workload of long documents tolerates a much larger block size than a workload of one-line chat replies. "16 is the default" is a statement about typical traffic, not a law.

---

# PART 3 — Sharing

## Refcounting and prefix sharing

The observation: in real serving, many concurrent sequences begin with **byte-identical** prefixes. A system prompt shared by every request. A few-shot example block prepended to every query. A RAG document reused across follow-ups. Multiple beams or samples branching from one prompt.

Identical prefix means identical K and V tensors — attention's K and V for position `i` depend only on the tokens at positions `0..i`, so if two sequences agree on their first `n` tokens they agree on the KV for those tokens exactly. Under naive allocation, `N` sequences each store their own copy of that identical data: `N` times the memory for one thing.

With a block table, sharing is nearly free. Both sequences' tables just list the *same* physical block id at the same logical index. No copying, no special case in the address arithmetic — translation already went through a table, and two tables can name the same block.

The one thing you now need is a rule for when a block may be released. A **reference count** is an integer per block recording how many sequences currently point at it. Free the block when it reaches zero, not when any one holder finishes. This is the same discipline as `shared_ptr` or CPython's own object refcounts.

```python
def incref(self, block_id: int) -> int:
    self._refcounts[block_id] += 1
    return self._refcounts[block_id]

def decref(self, block_id: int) -> int:
    """Drop one reference, freeing the block at zero. Returns the new count."""
    self._refcounts[block_id] -= 1
    if self._refcounts[block_id] == 0:
        self.free(block_id)
        return 0
    return self._refcounts[block_id]
```

Measured, Step 9: **8 sequences sharing a 64-token prompt at `block_size = 16`.**

```
prefix_blocks       = 64 / 16 = 4 blocks
naive (N copies)    = 8 × 4  = 32 blocks
actual (shared)     =          4 blocks
blocks saved        =         28        ← prefix_blocks_saved
max_refcount        =          8        ← max_refcount
```

**Gate S3.3** required `max_refcount == N` and `saving == (N−1) × prefix_blocks`. Both exact: `8 == 8`, and `28 == 7 × 4`.

The count of 8 is worth a second look, because it is easy to get off by one. The template sequence is not scaffolding — it is a live, referenced sequence, sharer number 1. Seven forks are created from it. `1 + 7 = 8`. If you had counted only the forks you would have expected 7 and called a correct implementation broken.

**How much does this actually save in production?** The saving is `(N−1) × prefix_blocks`, so it scales with both the number of sharers and the prefix length. Concretely: a 1000-token system prompt at `block_size = 16` is 63 blocks; 32 concurrent requests sharing it save `31 × 63 = 1953` blocks = 31,248 token-slots. Using Lab 1's figure of 128 KB/token for Llama-3.1-8B, that is roughly 3.9 GB of KV cache that does not need to exist — on a 24 GB machine. This is why prefix caching is one of the highest-leverage features in a real serving stack, and why it is the mechanism most write-ups about PagedAttention skip entirely in favour of the fragmentation story.

## Copy-on-write

Sharing is safe exactly as long as nobody writes. The moment one sharer needs to put its own token into a block that others also reference, that block must be duplicated first — for that sharer only — or one sequence's token appears in another sequence's cache. Silent, and catastrophic: the other sequence attends over a token it never produced.

This is `fork()` copy-on-write, and the parallel is worth stating properly because it is structurally the same, not merely similar. `fork()` gives the child a page table pointing at the parent's physical pages, marks them read-only, and increments each page's reference count. Neither process copies anything. On the first write, a protection fault fires, the kernel allocates a fresh page, copies the contents, points the writer's page table at the copy, and drops the old page's refcount by one. **One page copied, not the whole address space, and only the page that was written.**

`ensure_writable` is that, by hand:

```python
def ensure_writable(self, logical_idx: int) -> int:
    """COW: if refcount > 1, copy to a fresh block, decref the old, return the new id."""
    old_id = self.physical_for_logical(logical_idx)
    if self._allocator.refcount(old_id) <= 1:
        return old_id
    # Allocate before decref. Dropping the reference first could free the block and
    # then hand the very same physical block straight back as the "copy".
    new_id = self._allocator.allocate()
    self._blocks[logical_idx] = new_id
    self._allocator.decref(old_id)
    return new_id
```

Three things to notice in nine lines.

**The refcount ≤ 1 fast path.** If nobody else holds this block, there is nothing to protect and no copy is made. Unshared sequences pay nothing for the mechanism existing.

**Allocate before decref.** The comment names a genuine trap. If you decref first and the count hits zero, the block is returned to the free list — and `allocate()` would then hand you back the identical physical block as the supposed copy. You would have "copied" a block onto itself and silently kept sharing.

**Only that one block moves.** Not the sequence's history, not the other logical indices, not the other sharers' tables.

Step 10 hand-traces the prediction before running it, which is the right order. Before: the prefix's last logical block has refcount 8. One fork writes. After: that fork holds a **new** physical id at that index with refcount 1; the old block's refcount is 7; every other fork's table is byte-identical to what it was. All three asserted, including the one that would catch the worst bug:

```python
other_fork_ids = [f.physical_for_logical(last_shared_idx) for f in forks[1:]]
assert all(bid == old_block_id for bid in other_fork_ids), "every OTHER fork must be untouched"
```

## Defect F2 — the alignment bug

This is the best thing in the notebook, and it is a bug the project found in its own code. Reported as **F2** in the progress ledger: *"`BlockTable.fork_from` non-block-aligned KV corruption: CONFIRMED and FIXED."*

> [!note] Read the current `blocks.py` knowing it is the **fixed** version.
> `fork_from` today shares only complete blocks and its docstring explains why at length. The bug below is history, preserved because the reasoning is more instructive than the fix. The regression tests that pin it are `test_fork_mid_block_shares_only_complete_blocks` and `test_non_aligned_fork_does_not_collide_under_ordinary_growth`.

**The original design.** `fork_from(other, n_shared_tokens)` shared every block the prefix touched — including a partially filled tail block. The reasoning was that copy-on-write would protect it: if either sequence ever wrote into a shared block, `ensure_writable` would copy first. Sound, given a premise.

**The premise is false.** Nothing calls `ensure_writable` during ordinary growth. Look again at `append_token`: its only decision is `n_tokens % block_size == 0`. It has no idea whether the tail block is shared, and it never asks. **A write that looks like appending does not look like an edit**, so no copy-on-write is triggered — the trap that fires in `fork()` has no counterpart here, because there is no protection bit and no fault.

**The repro.** `block_size = 4`, parent grown to 6 tokens, fork at position 6, then append one token to each.

```
parent, 6 tokens, block_size 4:
    block 0 = [t0 t1 t2 t3]   FULL
    block 1 = [t4 t5 __ __]   PARTIAL -- 2 of 4 slots used

buggy fork(n_shared_tokens=6): shares both blocks
    parent.blocks == (0, 1)
    child.blocks  == (0, 1)          <-- block 1 shared while partial

parent.append_token()   ->  n_tokens 6 % 4 != 0  ->  no new block  ->  writes (block 1, offset 2)
child.append_token()    ->  n_tokens 6 % 4 != 0  ->  no new block  ->  writes (block 1, offset 2)

parent.slot_for_position(6) == (1, 2)
child.slot_for_position(6)  == (1, 2)      <-- THE SAME PHYSICAL SLOT
```

I reproduced this directly, against a local buggy re-implementation of `fork_from` in a scratch file (never touching the lab repo): both tables resolve position 6 to `(1, 2)`. The fixed code gives `(1, 2)` for the parent and `(2, 2)` for the child.

Both sequences believe they own the free slots in that tail block, because both were told they own the block, and "how many slots are still free in my tail" is computed from `n_tokens`, which each tracks privately. Then whichever writes second overwrites the other's K and V.

**What the failure looks like from outside.** Nothing. No exception, no assertion, no `OutOfBlocks`. Refcounts stay internally consistent. `wasted_slots` reports plausible values. Utilisation reports 0.875. One sequence simply attends over a token another sequence produced, and generates a fluent, wrong continuation. This is the failure mode the entire lab is built to avoid: a plausible-looking wrong answer with no signal attached.

**Why the notebook's own run never hit it.** Step 9 uses:

```
SHARED_PROMPT_TOKENS = 64
BLOCK_SIZE           = 16
64 / 16 = 4 blocks exactly, remainder 0
```

There *was* no partial tail block. The shared prefix ended precisely on a block boundary, so the buggy path and the correct path do the identical thing. Confirmed by re-running Step 9 after the fix: `max_refcount = 8`, `prefix_blocks_saved = 28`, unchanged.

So the notebook's measurement was correct — **by luck.** Sixty-four is a round number, chosen because round numbers make demonstrations legible, and it happened to be a multiple of the block size. Had the shared prompt been 60 tokens, the same cell would have quietly corrupted its own KV and reported the same healthy-looking numbers.

> A green demonstration is evidence about the case you demonstrated, not about the mechanism. Ask what the aligned, round, or convenient parameter in your setup is hiding — and test the awkward value on purpose.

**The fix.** Share only complete blocks; give the fork a fresh, exclusively owned block for any partial remainder.

```python
n_complete_shared_blocks = n_shared_tokens // self._block_size
for logical_idx in range(n_complete_shared_blocks):
    block_id = other.physical_for_logical(logical_idx)
    self._allocator.incref(block_id)
    self._blocks.append(block_id)
self._n_tokens = n_complete_shared_blocks * self._block_size

remainder = n_shared_tokens - self._n_tokens
if remainder > 0:
    self._blocks.append(self._allocator.allocate())
    self._n_tokens += remainder
```

Two details. `//` instead of `ceil` is the whole correctness change. And because this module holds no tensors — it tracks which physical block id belongs to which logical position — "copy the remainder" here means claiming a fresh block for it; a caller with real KV tensors copies the actual K/V values into the new block using that id. The docstring says so explicitly rather than leaving a reader to assume the data moved.

This is also what vLLM's prefix caching does, and for this exact reason. Block-granular sharing is a **correctness requirement**, not an implementation convenience.

**The general lesson, stated at the right altitude:**

> Your sharing granularity must equal your write granularity. If you share in units of blocks but write in units of tokens, then a unit exists that is shared and writable at once — and every write into it is a race unless something intercepts writes at token granularity. You get two choices: refuse to share partial units, or implement copy-on-write below the sharing unit. There is no third option, and "copy-on-write will handle it" is not an answer unless something actually calls it on the path that writes.

The test that had existed all along did not catch this, and *why* it did not is the last part of the lesson: it called `ensure_writable` manually before writing. It tested the mechanism in the manner the author was thinking about, not in the manner a caller would actually use. **Test the natural usage path, not the correct-if-you-remember one.** The replacement regression test calls plain `append_token()` on both tables and asserts the slots differ — which is precisely what real code does.

---

# PART 4 — The boundary

## What was built, and what was not

This section is not modesty. Being exact about where the work stopped is a stated value of this lab, and it is the difference between a claim that survives questioning and one that does not.

**Built, real, measured, unit-tested.** `BlockAllocator`'s pool and reference counting, `BlockTable`'s logical-to-physical map, incremental allocation, prefix sharing via `fork_from`, copy-on-write via `ensure_writable`, release-and-reuse. `tests/test_blocks.py` at 29/29 passing; the repo at 122/122 with `ruff` clean. Every measurement above — 0.108, 0.875, the sweep, refcount 8, 28 blocks saved — is a property of that code, not a simulation of code that might exist.

**Not built: the fused attention kernel.** A production PagedAttention implementation writes a custom attention kernel that, for each sequence, reads the block table and gathers that sequence's scattered physical blocks *inside* the attention computation — no separate materialisation step, no contiguous scratch buffer. That kernel does not exist here.

So the honest description is: **this is PagedAttention's memory manager, not PagedAttention.**

Why is the kernel the hard part?

- **It lives in the innermost loop.** Attention over a sequence of length `n` touches every one of `n` cached positions, every layer, every step. Any indirection you add is paid `n × L` times per token. This is where all the performance is, and it is unforgiving.
- **It must not lose coalescing.** The whole reason contiguity was attractive is wide, coalesced memory reads. A gather-based kernel has to reorganise its access pattern so that within a block reads are still contiguous, and arrange blocks so the boundaries do not stall the pipeline. That is the actual engineering content of the PagedAttention paper.
- **The platform here does not offer a path.** vLLM's kernels are CUDA. This lab runs on Apple Silicon via MLX, where the equivalent means writing a custom Metal kernel from scratch — a different and much larger project than the allocator, and one this lab does not attempt.

The important thing to hold onto: **the memory win is genuinely in the allocator, not the kernel.** The kernel is what makes the allocator's layout *fast enough to use*. Utilisation going from 0.108 to 0.875, and `N` copies of a system prompt going to 1, are allocation results. The kernel is the tax you must pay to collect them at speed.

The notebook's own sentence for this, worth borrowing verbatim: *"I built the allocator and measured the fragmentation win; the kernel-side gather is where a CUDA implementation earns the rest."*

## Step 12 — scoped out, and why

`step12_scoped_out: true` in the artifact refers to a step the LLD listed as **conditional**, marked *"(if S0 permits)"*: **generate one real sequence through the block table** — gather the scattered blocks before each attention call, verify the token IDs match unpaged generation at temperature 0, and report the slowdown. It was not attempted.

The reason given is specific, and it is not "ran out of time."

`mlx_lm`'s native `KVCache` stores each layer's keys and values as one contiguous array. Feeding attention from non-contiguous physical blocks instead means writing a custom from-scratch cache class, per layer, that gathers scattered blocks into a contiguous scratch buffer before every attention call. Then you must prove — token for token, at temperature 0 — that it produces exactly what the native contiguous path produces.

The lab's own required reading, `MLX_IN_PRACTICE.md` §5, warns about precisely this shape of change: *"two code paths that are supposed to be numerically equivalent are not, and the divergence is occasionally large enough to send a long generation somewhere pathological... treat any batched or cached path as a separate implementation requiring its own correctness evidence."* This is not a hypothetical for this project — defect F1, a batched-decode divergence on the same stack, was investigated at the logit level and written up in `artifacts/S0_spike_verdict.md`.

So the argument is: Steps 5–10 already establish, measured, what a paged block table is and what it saves. Step 11 already states precisely which half of PagedAttention that is. An unvalidated gather-based cache on top would not strengthen either claim — it would add a second place a silent numerical divergence could hide, which is the exact failure mode the lab exists to avoid. Scoped out and logged as open work rather than attempted and hoped correct.

The open question left behind is a good one, and it names the smaller experiment: *before ever sampling a token, check whether gathered K/V tensors are bit-identical to the contiguous cache's tensors for a handful of steps.* That is a cheap, decisive test of the gather logic alone, with no generation loop to hide a discrepancy in.

> Declining a step with a written reason is a result. "Not attempted" plus the argument is stronger than a number you cannot defend.

---

# Recall checklist

Blank sheet, no lookups:

- [ ] Why is max-length reservation the *natural* thing to write? Name both constraints that push you there, and say which one paging attacks.
- [ ] What was mean utilisation under max-length reservation, and what were the numerator and denominator? Why was the 512 ceiling a charitable choice rather than a rigged one?
- [ ] What does "oracle" mean here, why is it exactly 1.0, and what does comparing against it let you say that you otherwise could not?
- [ ] Define internal and external fragmentation in one sentence each. Which one does right-sizing every reservation fix, and which one does right-sizing make *worse*?
- [ ] Given `block_size` and a token position, write the two lines that produce `(physical_block, offset)`. Then state which half is per-sequence state and which is pure arithmetic.
- [ ] Write `append_token` from memory. What is the exact condition under which it allocates, and why is that one line the whole mechanism?
- [ ] Paging bounds internal fragmentation by what quantity — and what did that waste scale with *before*? Why is external fragmentation eliminated rather than merely reduced?
- [ ] Derive, do not recall, why `block_size = 1` gives utilisation exactly 1.0. Then give three distinct reasons nobody ships it.
- [ ] Why 16? Point at the knee in the sweep and say what is being traded on each side. What property of the *workload* would justify a larger value?
- [ ] Distinguish `peak_allocated` from `total_blocks_ever`. Which one sizes your hardware? What would it mean if they were equal?
- [ ] The external-fragmentation gate had two clauses. What was the second, and what would a one-clause gate have reported on a run with 2 failures of which 1 was scarcity?
- [ ] Sketch the state before and after one forked sequence writes into a shared block: refcounts, block ids, and what happens to the other sharers.
- [ ] In `ensure_writable`, why must you allocate before you decref? Describe the bug if you reverse them.
- [ ] Reproduce defect F2 on paper: `block_size = 4`, 6 tokens, fork at 6, one append each. Which physical slot collides, and why did nothing raise?
- [ ] Why did the notebook's own Step 9 never hit F2? What was the fix, and state the general rule about sharing granularity versus write granularity.
- [ ] Which half of PagedAttention is built here? Explain why the fused gather kernel is the hard half, and why the memory win is nevertheless in the allocator.
- [ ] What was Step 12, and give the actual reason it was scoped out. What smaller experiment would test the same thing at lower risk?
