




## 0) The one-sentence map (keep this in your head)

A Transformer is a repeated alternation of 
**(A) content-addressed read from a sequence memory (attention)** and 
**(B) token-wise nonlinear computation (MLP/FFN)**; 

modern LLM efficiency is dominated by (1) the _quadratic_ cost of (A) at long context and (2) the _bandwidth/memory_ of storing attention’s past state (KV cache), so the big tricks are: **share/compress KV (MQA/GQA), compress KV even further via a latent bottleneck (MLA), reduce compute via conditional FFN (MoE), and finally reduce long-context attention itself via sparse selection (DSA).**

---

## 1) From seq2seq to attention: why “QKV” exists at all

### 1.1 The original seq2seq bottleneck

Classical seq2seq (RNN encoder → fixed vector → RNN decoder) had a core failure mode: a single fixed-size vector must carry all source information. As sequences grew, the decoder lost details. Attention was invented to replace the fixed bottleneck with a **dynamic retrieval mechanism**: for each output step, the decoder “looks back” into all encoder states and pulls out the relevant pieces.

### 1.2 The attention primitive as “content-addressed memory”

Think of the source states as a memory with L slots. Each slot has:

- a **key**: “what is stored here?” (addressing)
- a **value**: “what do I get if I retrieve this slot?” (content)

The decoder step produces a **query**: “what am I looking for now?”

Then attention does: compare query to each key, turn similarities into weights, take a weighted sum of values. That’s the entire conceptual leap.

In Transformer notation (self-attention is the same idea but source = the sequence itself):

- Hidden states: `X` with shape `[L, d_model]`
- Linear maps: `W_Q, W_K, W_V`
- `Q = X W_Q`, `K = X W_K`, `V = X W_V`
- Weights: `A = softmax(Q K^T / sqrt(d_k))`
- Output: `O = A V`

The scaling `1/sqrt(d_k)` is not decoration; it keeps dot products in a numerically stable range as dimension grows.

### 1.3 Why split into heads?

Single-head attention forces one similarity metric and one retrieval mixture. Multi-head attention says: learn multiple, parallel retrieval systems. Each head has its own projections; heads attend differently (syntax-like vs entity-like vs positional patterns), then get concatenated and mixed back.

Mechanically:

- `h` heads, each head dimension `d_head = d_model / h`
- Each head i has `W_Q^i, W_K^i, W_V^i`
- Outputs are concatenated: `O = concat(O^1..O^h) W_O`

Mental model: **each head is a different learned “lens” for addressing and reading memory**.

---

## 2) Autoregressive decoding and the KV-cache: the real bottleneck in production

### 2.1 Why caching exists

In autoregressive generation, at step t you only produce one new token, but attention wants to compare the new query against all past keys. If you recompute K and V for all previous tokens every step, decoding becomes catastrophically redundant.

So you cache. At each layer, you store:

- `K_cache[layer]` containing keys for tokens 1..t
- `V_cache[layer]` containing values for tokens 1..t

Then at step t+1 you compute only:

- the new `q_{t+1}`, new `k_{t+1}`, new `v_{t+1}`
- append `k_{t+1}, v_{t+1}` to the cache
- attend `q_{t+1}` against `K_cache` (length t+1)

This reduces compute per step from “rebuild everything” to “one new projection + one attention against stored keys.”

### 2.2 The hidden cost: KV-cache is huge

In big LLMs, decoding is often bandwidth-bound because KV cache dominates memory traffic.

Rough scaling per layer (ignoring constants):

- store K and V for each token: `O(L * n_heads * d_head)`
- across layers: multiply by `n_layers`
- for batch size B, multiply again by B
- precision matters (FP16/BF16 vs FP8)

So as L grows, KV cache becomes the _limiting resource_ for throughput and max context.

This is where MQA/GQA and then MLA enter: **they attack KV cache size and bandwidth**.

---

## 3) MHA vs MQA vs GQA: the clean conceptual progression

### 3.1 Standard multi-head attention (MHA)

Each head has its own K and V. That is expressive, but it’s expensive to store: you store K/V per head, per layer, per token.

If you have 32 query heads, you also have 32 KV heads. Cache is proportional to 32.

### 3.2 Multi-Query Attention (MQA): “many Q heads, one shared KV”

MQA keeps many query heads (so you keep multiple “ways to ask”), but **shares K and V across all query heads**:

- `n_q_heads` large
- `n_kv_heads = 1`

What changes?

- You still compute multiple Q heads (cheap relative to storing long histories).
- But you store only one K and one V per token (per layer), not per head.

So KV cache shrinks by about `n_q_heads / n_kv_heads` (e.g., ~32x if truly one KV head). That’s enormous for long context.

Why doesn’t it completely ruin quality? Because queries can still differ; they all read from the same memory representation, but they weight it differently. You lose some per-head specialization in “what counts as a key/value,” but keep diversity in “how you query and mix.”

MQA’s big win is not FLOPs; it’s memory/bandwidth and kernel simplicity.

### 3.3 Grouped-Query Attention (GQA): the quality-preserving compromise

GQA is a midpoint:

- `n_q_heads` large
    
- `n_kv_heads` > 1 but smaller than `n_q_heads`
    

Example: 32 Q heads, 8 KV heads. Each KV head is shared by a group of 4 Q heads. Cache shrinks by ~4x relative to full MHA, but preserves more diversity than MQA.

Interview-level line: **MQA/GQA are “KV-sharing schemes” that reduce cache size while retaining multi-head query diversity.**

---

## 4) “Latent KV” as a deeper compression idea: why MLA exists

MQA/GQA share K/V across heads. But you still store K and V vectors for every token. MLA goes one level deeper: **don’t store K/V vectors at all; store a smaller latent that can regenerate them.**

### 4.1 The general latent-KV pattern (first principles)

Suppose we can factor K and V through a low-dimensional bottleneck:

- Instead of storing `K_t` and `V_t` directly, store `c_t` where `dim(c_t) << dim(K_t)+dim(V_t)`.
    
- At attention time, reconstruct approximate keys/values via learned projections:
    
    - `K_t ≈ f_K(c_t)`
        
    - `V_t ≈ f_V(c_t)`
        

If the bottleneck retains the information attention needs, you get a huge cache reduction.

This is exactly the same idea as low-rank approximations in linear algebra: store the “essence” and reconstruct.

### 4.2 What MLA (“Multi-Head Latent Attention”) is doing conceptually

DeepSeek’s MLA is best understood as:

1. Map each token hidden state `h_t` into **two small latent vectors** (one for queries, one for KV).
    
2. Use those latents plus small per-head projections to construct the actual `q, k, v` used by attention.
    
3. Cache the KV latent (or a compressed representation) rather than full per-head KV.
    

So MLA is “latent KV + multi-head reconstruction.”

The high-value intuition: **MLA reduces the stored state of the model’s attention memory**; attention becomes a function of a compact latent cache rather than a full KV cache.

### 4.3 Why this matters more than it sounds

In production decoding, you are often limited by:

- how many concurrent sequences you can keep alive (batch size),
    
- how long they can be (context),
    
- how fast you can stream outputs (bandwidth).
    

Reducing KV cache size directly increases either:

- maximum context length,
    
- batch size at same context,
    
- throughput at same batch/context,
    
- or all three (depending on scheduler).
    

This is why “KV-cache compression” is not a minor optimization; it is a first-order lever on serving cost.

### 4.4 How RoPE and positional encoding interacts with these designs

Rotary position embeddings (RoPE) are applied to Q and K (not V). Any design that compresses or reconstructs K must preserve positional structure. Many modern attention variants effectively split K into “content part” and “positional part” (or apply RoPE to only a subset). When you see diagrams showing “apply RoPE here but partially there,” that’s an implementation of the principle: **you can compress content aggressively, but position-sensitive components must remain structurally compatible with RoPE.**

This is why MLA diagrams often show partially-applied RoPE and separate components.

---

## 5) MoE in one coherent mental model: compute sparsity, not memory sparsity

MoE swaps the dense FFN (MLP) with many experts:

- A router reads the token representation and picks top-k experts.
    
- Only those experts run for that token.
    
- You get huge total parameter count, but only a fraction is “active per token.”
    

So MoE changes the scaling axis:

- Dense model: more parameters generally means more FLOPs per token.
    
- MoE model: more parameters can mean similar FLOPs per token (because you only activate few experts).
    

Key production truth: MoE’s primary pain is **routing and communication** (all-to-all dispatch) and ensuring balanced expert usage. But the reason it’s attractive is simple: **it increases model capacity per unit compute**.

Now combine with attention: attention is often memory/bandwidth constrained; MoE targets compute. They are complementary.

---

## 6) Reconstructing the DeepSeek stack: MoE + MLA + DSA, and why DSA is “under MLA” and often uses MQA mode

You highlighted three statements; here is the tight reasoning chain that makes them feel inevitable.

### 6.1 “DeepSeek’s attention stack already uses MLA; V3.2 adds sparsity on top”

Interpretation:

- MLA is already compressing the KV cache and making attention memory-friendly.
    
- But even with a smaller cache, **dense attention over long contexts still costs too much compute** (prefill is quadratic; even decode has big constant factors when context is huge).
    
- So V3.2 adds DSA to reduce _which tokens_ are attended to, not just how KV is stored.
    

MLA solves: “KV cache is too big.”  
DSA solves: “Even reading the cache densely is too expensive at long context.”

That’s why it’s natural to say “DSA on top of MLA.”

### 6.2 “The backbone (MoE + MLA) stays; attention gets a new sparse path”

Interpretation:

- MoE and MLA are already huge engineering investments, and they target orthogonal bottlenecks:
    
    - MoE: reduce compute per token for FFN capacity
        
    - MLA: reduce memory/bandwidth for attention cache
        
- DSA is an incremental bolt-on that changes attention’s _selection pattern_ but can reuse MLA’s internal representations.
    

So you keep the backbone to preserve stability, reuse checkpoints, and avoid retraining from scratch, while adding a learned selector that gates which KV entries are used.

### 6.3 “DSA under MLA and implemented in MQA mode”

This line becomes clear once you view DSA as a pipeline with two stages:

Stage A: **Indexing/scoring** (“Lightning Indexer”): for each query, produce a relevance score over past tokens.  
Stage B: **Core attention**: attend only over top-k selected KV entries.

Now ask: what is the easiest way to implement Stage B efficiently, without destroying kernel performance?

If you use full MHA with many KV heads, then “top-k selection” becomes tricky:

- Do you select a different top-k per head? That multiplies complexity and makes memory access patterns ugly.
    
- If selection is shared across heads, you still have to gather many KV tensors.
    

MQA makes this simpler and cheaper:

- With shared KV, you can select top-k tokens once (or per small group) and gather a single KV stream.
    
- All query heads attend to the same selected KV entries, differing only by Q.
    
- This reduces gather/scatter overhead and makes kernels far more stable under batching.
    

There’s also a checkpoint-continuation logic: if the existing model already uses a KV-sharing or latent-KV scheme aligned with MQA-style attention kernels, implementing DSA in that mode reduces architectural discontinuity and makes continued training easier.

The deep reason is: **DSA introduces a discrete selection step; discrete steps and high-head-count KV layouts don’t mix nicely for throughput**. MQA/GQA (and MLA’s latent structure) produce a KV representation that is easier to index and gather.

---

## 7) “Permanent memory” hooks: how to remember all of this without re-deriving every time

Here are a few durable mental anchors. They are not slogans; they are compression algorithms for your brain.

### 7.1 QKV as retrieval

Q = what I want, K = where it is, V = what I get.  
Attention = “soft address → weighted read.”

If you can say this smoothly in interviews, you’re already ahead of most candidates.

### 7.2 Heads as parallel retrieval subspaces

Multiple heads do not mean “more attention”; they mean **multiple learned similarity spaces and multiple retrieval mixtures**.

### 7.3 KV cache is the state of the decoder

During autoregressive decoding, the _entire past_ lives inside KV cache per layer. That is why serving cost is dominated by KV cache memory traffic.

### 7.4 MQA/GQA are KV-sharing

MQA: one KV for all Q heads.  
GQA: a few KV groups shared across Q heads.  
Purpose: shrink KV cache and simplify kernels; quality trade-off is controlled by number of KV groups.

### 7.5 MLA is “store latents, reconstruct KV”

Instead of caching full KV vectors, cache a smaller latent (compressed memory), and reconstruct keys/values as needed. This attacks the KV cache bottleneck more fundamentally than MQA/GQA.

### 7.6 MoE is compute sparsity

Only a few FFN experts run per token. This increases capacity per FLOP, but doesn’t directly fix attention’s long-context cost.

### 7.7 DSA is attention sparsity

Only a small set of tokens are attended to (top-k) per query. This attacks the quadratic cost of dense attention in long contexts.

### 7.8 Why DSA likes MQA under MLA

Selection is a gather/index operation; gather/index wants a simple KV layout. KV-sharing and latent-KV layouts make sparse selection tractable and fast.

---

## 8) Interview-ready mini-explanations you should be able to produce (verbatim-style)

If you want “20–50% reproducible,” aim for these three answers.

First: “What is attention?”  
Attention is content-addressed retrieval: queries compare to keys to produce weights, then read a weighted sum of values. In Transformers it’s self-attention, so the sequence is its own memory; multi-head means multiple retrieval subspaces in parallel.

Second: “What is KV cache and why is it a bottleneck?”  
In autoregressive decoding, we cache keys and values for every past token at every layer so we don’t recompute them each step. That cache scales with layers × sequence length × head dimensions, so at long context it dominates memory bandwidth and limits throughput and batch size.

Third: “What are MQA/GQA/MLA, and why combine them with sparse attention?”  
MQA/GQA share KV across heads to shrink cache; MLA goes further by caching a compact latent representation and reconstructing KV on the fly, reducing bandwidth and memory. But dense attention still scales poorly with long prompts, so sparse attention like DSA uses a learned indexer to pick top-k relevant past tokens, making attention closer to O(L·k). Implementing sparse selection is much easier when KV is shared or latent-compressed, which is why DSA sits naturally under MLA and often uses MQA-style KV layouts for kernel efficiency.

---

----


# Notebook Lesson A — Re-deriving attention + KV cache with real shapes

### A0. Fix a concrete model shape (I keep these constants across the whole notebook)

I pick a common “LLM-ish” setting:

- `d_model = 4096`
    
- number of query heads `Hq = 32`
    
- so `d_head = d_model / Hq = 4096 / 32 = 128`
    
- sequence length for examples: I’ll use both `L = 4096` (small) and `L = 128k` (long-context stress test)
    
- dtype for cache: assume BF16/FP16 → `2 bytes` per element (later I’ll also mention FP8)
    

I write this at the top because every later memory/computation estimate depends on it.

### A1. Single-layer self-attention, one token position t

Input hidden states: `X` has shape `[L, d_model] = [L, 4096]`.

Linear projections:

- `Q = X W_Q`, `K = X W_K`, `V = X W_V`
    
- In “full MHA,” each is shaped to heads:
    

`Q` reshapes to `[L, Hq, d_head] = [L, 32, 128]`  
`K` reshapes to `[L, Hk, d_head]` (for MHA, `Hk = 32`)  
`V` reshapes to `[L, Hv, d_head]` (for MHA, `Hv = 32`)

At one position `t`, I take:

`q_t` shape `[Hq, d_head] = [32, 128]`  
`K_{<=t}` shape `[t, Hk, 128]`  
`V_{<=t}` shape `[t, Hv, 128]`

Causal attention per head:

For head `h`, scores are dot products:

`s_{t,j,h} = <q_{t,h}, k_{j,h}> / sqrt(128)` for `j = 1..t`

So per head I compute `t` dot products of length 128, then softmax over `j`, then weighted sum of values.

Output per head:

`o_{t,h} = Σ_j softmax(s_{t,*,h})_j * v_{j,h}` → shape `[128]`

Concatenate heads:

`o_t = concat_h(o_{t,h})` → shape `[4096]` → then multiply by `W_O`.

This is the “pure math.”

### A2. Autoregressive decoding: what gets recomputed vs cached

In decoding, at time step `t` I already computed keys/values for tokens `1..t-1` in previous steps.

So I cache:

`K_cache` per layer: keys for all previous tokens  
`V_cache` per layer: values for all previous tokens

At step `t`, I only compute `k_t, v_t` for the new token, append them, and compute attention using cached K/V.

This is the critical engineering move: compute scales with “new token + attention read,” not “rebuild everything.”

Now the important part that makes MQA/GQA/MLA feel real:

**Even if compute is okay, storing K/V for all tokens across all layers is huge.** That’s the KV-cache problem.

---

# Notebook Lesson B — KV cache math you can do in your head (MHA → GQA → MQA)

I’m going to write KV cache size formulas in a way you can reproduce on a whiteboard.

### B0. One layer KV cache elements per token

Per token, the cache stores K and V.

- K elements per token = `Hk * d_head`
    
- V elements per token = `Hv * d_head`
    

So total KV elements per token per layer:

`E_token_layer = (Hk + Hv) * d_head`

In almost all implementations, `Hk = Hv = H_kv` (same number of KV heads), so:

`E_token_layer = 2 * H_kv * d_head`

Multiply by bytes per element (2 bytes for BF16/FP16):

`bytes_token_layer = 2 * H_kv * d_head * 2`

That extra `*2` at the end is “K plus V,” and the earlier `*2` is “2 bytes.” Easy to mix up, so I keep them separate.

### B1. Plug numbers for MHA (H_kv = Hq = 32)

For MHA: `H_kv = 32`, `d_head = 128`.

Elements per token per layer:

`E_token_layer = 2 * 32 * 128 = 2 * 4096 = 8192 elements`

Bytes per token per layer (BF16):

`bytes_token_layer = 8192 * 2 = 16384 bytes ≈ 16 KB`

That “16 KB per token per layer” is the number to remember.

Now multiply by layers and sequence length.

If `N_layers = 60` (representative for large LLMs), per token across layers:

`16 KB * 60 = 960 KB per token` (almost 1 MB per token)

So for long context `L = 128k` tokens:

`960 KB/token * 128,000 tokens ≈ 122,880,000 KB ≈ 117 GB`

That’s just the KV cache, not weights, not activations. This is why long-context serving is hard.

Even at `L = 32k`, it’s about a quarter: ~29 GB KV cache at 60 layers.

This is the core “why KV compression exists” argument, made numeric.

### B2. GQA example: Hq = 32, H_kv = 8

GQA says: keep 32 query heads, but only 8 KV heads.

Then:

`E_token_layer = 2 * 8 * 128 = 2048 elements`

Bytes per token per layer:

`2048 * 2 = 4096 bytes = 4 KB`

Compare to MHA: 4 KB vs 16 KB → **4x smaller KV cache**.

Now redo the earlier long-context estimate:

Per token across 60 layers:

`4 KB * 60 = 240 KB per token`

At `L = 128k`:

`240 KB * 128,000 = 30,720,000 KB ≈ 29.3 GB`

So GQA(8) turns the earlier ~117 GB into ~29 GB. That’s already the difference between “impossible” and “maybe feasible on one GPU” depending on the setup.

### B3. MQA example: Hq = 32, H_kv = 1

MQA says: one KV head shared by all Q heads.

Then:

`E_token_layer = 2 * 1 * 128 = 256 elements`

Bytes per token per layer:

`256 * 2 = 512 bytes`

Compare to MHA: 512 B vs 16 KB → **32x smaller KV cache**.

Across 60 layers:

`512 B * 60 = 30,720 B ≈ 30 KB per token`

At `L = 128k`:

`30 KB * 128,000 ≈ 3,840,000 KB ≈ 3.7 GB`

This is why MQA is so important for extreme contexts and high throughput: it makes KV cache manageable.

### B4. The “why quality doesn’t collapse” intuition (the part you say in interviews)

In MHA, each head has its own K/V space; in MQA, all heads share one K/V space but keep separate Q projections. So you retain diversity in “how you ask,” and you lose diversity in “how memory is represented.” In practice, that trade often pays off because Q diversity is enough to recover many behaviors, and the serving gains are enormous.

---

# Notebook Lesson C — MLA as “latent KV”: same game, one level deeper

MQA/GQA share KV heads. MLA attacks the next bottleneck: even one KV head can be large over 128k tokens and 60 layers, and the bandwidth of moving KV around matters.

I write the generic latent KV trick:

Instead of caching K and V directly, cache a latent `c_t` with `d_c << (H_kv * d_head)` and reconstruct K/V from it.

### C1. Make the latent explicit

Let `c_t` be cached with dimension `d_c = 512` (toy but plausible as a bottleneck scale; the exact value depends on the paper’s design).

Then per token per layer cached elements = `d_c`.

Bytes per token per layer = `d_c * 2` (BF16)

So if `d_c = 512`:

Bytes per token per layer = `512 * 2 = 1024 bytes = 1 KB`

Compare this to earlier:

- MHA: 16 KB/token/layer
    
- GQA(8): 4 KB/token/layer
    
- MQA(1): 0.5 KB/token/layer
    
- Latent KV with 512 latent: 1 KB/token/layer
    

So a 512-latent is not automatically smaller than MQA(1) on cache size alone. The point is: MLA isn’t just “one latent”; it typically also changes how many components are cached and how heads are reconstructed, and it can be paired with other design choices (partial RoPE application, shared KV structure) that reduce the effective cached payload and bandwidth.

The right “first principles” takeaway is:

**MLA creates a compressed, cache-friendly representation of attention memory. It turns “store exactly K/V” into “store a smaller sufficient statistic and reconstruct.”**

### C2. The reconstruction shapes (what I literally write down)

Cached: `c_{1..t}` has shape `[t, d_c]`.

Reconstruct keys/values (one conceptual variant):

`k_j = c_j W_k` where `W_k` has shape `[d_c, H_kv * d_head]`  
`v_j = c_j W_v` where `W_v` has shape `[d_c, H_kv * d_head]`

Then reshape `k_j` to `[H_kv, d_head]`.

This adds compute at attention time (a matmul from latent to KV), but often that compute is cheaper than moving/storing massive KV tensors, especially when memory bandwidth is the real limiter.

### C3. RoPE “partial application” becomes intuitive

RoPE is applied to Q and K. If K is reconstructed from a latent, it’s convenient to separate K into a part that gets RoPE and a part that doesn’t, or to apply RoPE after reconstruction to only the relevant components. This is what those diagrams are visually encoding: you can compress content heavily, but you must preserve a position-aware pathway compatible with RoPE.

---

# Notebook Lesson D — Why DSA “under MLA” and why MQA mode makes the sparse gather workable

Now we stack the ideas properly.

### D1. Dense attention cost vs sparse attention cost

Even if caching is compressed, dense attention still reads “too much” at long context.

For one query token:

- Dense attention compares against `t` keys → O(t * d_head) per head
    
- Across heads and layers, that’s expensive for very large t
    

Sparse attention says: select only `k` past tokens, attend over those.

So the core attention matmul becomes O(k * d_head) per head instead of O(t * d_head).

If `t = 128k` and `k = 2048`, the reduction in “tokens attended to” is:

`128,000 / 2,048 ≈ 62.5x`

That’s why a top-k selector is worth building.

### D2. The Lightning Indexer as “cheap attention teacher”

DSA introduces an auxiliary scorer that estimates which past tokens matter. The clean way to think of it is:

- The full attention pattern (from MLA core attention) is the “teacher.”
    
- The indexer is a “student” trained to approximate which tokens would have high attention mass.
    
- After it learns, you use it to choose top-k tokens and stop paying for dense attention.
    

This makes DSA not feel like a heuristic; it’s a learned approximation of where attention would go.

### D3. Why “under MLA”

If MLA is the internal attention mechanism (how Q/K/V are formed and cached), DSA’s job is upstream: it decides **which token positions** to include in the attention computation. That selection is naturally a wrapper around the attention module, not a replacement for how K/V are represented.

So “DSA under MLA” really means: DSA is implemented as a selection mechanism integrated into the MLA attention path, leveraging MLA’s cache representation and head structure.

### D4. Why MQA mode becomes attractive in sparse attention

Sparse attention requires a gather: you choose indices `I = {j1..jk}`, then gather `K_I, V_I`.

If you have many KV heads (full MHA), gathering becomes more complex: you’re gathering a larger tensor and the kernel has to handle head-specific layouts.

In MQA:

- KV is shared; gathering KV is simpler and smaller.
    
- All query heads reuse the same gathered KV; only Q differs per head.
    

So sparse selection + MQA tends to be a very clean systems pairing:

- One index set per token position.
    
- One KV gather per layer (not per head).
    
- Many Q heads attend to the same sparse KV set.
    

This is the kernel-efficiency story in one line: **DSA adds indexing and gathering; MQA makes that gather cheap and regular.**

---

# Practice-by-proxy worksheet (I write it as if I’m practicing)

### Exercise 1: KV cache size sanity check

I fix: `d_model=4096`, `Hq=32`, `d_head=128`, `layers=60`, `L=32k`, dtype BF16.

I compute MHA cache size:

Per token per layer: `2 * 32 * 128 = 8192` elems → `8192*2 = 16KB`  
Per token across layers: `16KB*60 = 960KB`  
For `32k` tokens: `960KB*32000 ≈ 30,720,000 KB ≈ 29.3 GB`

Then GQA(8): divide by 4 → about 7.3 GB.

Then MQA(1): divide by 32 → about 0.9 GB.

I write a margin note: “This is why MQA is a serving unlock.”

### Exercise 2: Sparse attention token reduction

I fix: `t = 128k`, `k = 2048`.

Reduction factor ≈ `62.5x`.

I write: “Even if the indexer itself has some overhead, there’s huge room for net win.”

### Exercise 3: The one interview sentence connecting them

I write: “KV-sharing (MQA/GQA) reduces cache size; latent KV (MLA) compresses the cache representation; sparse attention (DSA) reduces the number of cached tokens we actually read. MoE reduces FFN compute. Together they target the real bottlenecks in long-context inference.”

---

# Tiny recall checklist (for your next reread)

When you reread these notes tomorrow, force yourself to recompute, without looking, just these three numbers:

1. `d_head = 4096/32 = 128`
    
2. MHA KV cache per token per layer in BF16 ≈ `16 KB`
    
3. Top-k reduction: `128k / 2048 ≈ 62.5x`
    

If those three become automatic, the rest tends to “stick” because your brain has anchors.