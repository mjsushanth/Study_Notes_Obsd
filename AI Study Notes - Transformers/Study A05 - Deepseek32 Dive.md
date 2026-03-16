
![[Pasted image 20260212111653.png]]

Intelligence so cheap ?? 2x or more model tokens in test and still 10x times cheaper. 

![[Pasted image 20260212111731.png]]

3.1 terminus is already very elegant, 685b something MOE model.


![[Pasted image 20260212112102.png]]

![[Pasted image 20260212112112.png]]


1. The first is the **“Scaling GRPO”** section showing the **GRPO objective** (a PPO-style clipped objective over a _group_ of sampled responses, with a **group-normalized advantage** and a **KL penalty**).
2. The second is a **DeepSeek-V3 block diagram**: a Transformer block where the **FFN is MoE-routed** (router + top-k experts + shared expert), and attention is **MLA (Multi-Head Latent Attention)**.
3. The third is the key **DeepSeek-V3.2 DSA diagram**: **DSA instantiated under MLA**, where a **Lightning Indexer** scores past tokens and a **Top-k selector** chooses a sparse KV set for core attention.



## Lesson 1 — Why long-context Transformers get expensive (and what “sparse attention” really means)

A vanilla Transformer spends most of its marginal compute on **attention**, because attention forms an LxL interaction pattern: for sequence length L, each layer computes something like `softmax(QK^T)V`, and `QK^T` is the killer—its cost and memory scale roughly as **O(L^2)** (more precisely O(L^2 * d_head) for the matmul plus bandwidth costs). This is why “128K context” is not just “4x 32K,” it’s closer to **16x** the quadratic part, especially in the **prefill** stage (processing the prompt). DeepSeek’s V3.2 report explicitly frames the win as reducing core attention from **O(L^2) to O(L*k)** by selecting only k relevant tokens per query, with k much smaller than L.

Sparse attention is not one thing; it’s a family of strategies that all try to exploit the same empirical truth: **for any given query token, only a small subset of prior tokens matter**. The main design axes are:

**(A) Fixed-pattern sparsity (engineered masks).** Examples: sliding windows, dilated/strided patterns, block-sparse layouts (Longformer/BigBird-style ideas). These are easy to implement and fast, but they’re “blind”: you attend to tokens because they’re nearby or in a block, not because they’re relevant.

**(B) Content-based sparsity (dynamic selection).** You build a cheap mechanism that predicts which past tokens are relevant, then run full attention only over those. This is what DeepSeek calls **DSA**: a lightweight scorer (Lightning Indexer) + top-k selection, then attention on the selected KV set.

**(C) External retrieval (RAG).** You don’t try to attend within the whole context; you retrieve a smaller context from an index and feed that. RAG helps knowledge access and long documents, but it changes the system boundary (you now depend on retrieval infra) and doesn’t directly make the model’s _internal_ long-context attention cheaper.

**(D) KV/cache compression and sharing (MQA/GQA, low-rank KV, latent KV).** This reduces memory and bandwidth. DeepSeek’s attention stack already uses **MLA**, which is a form of “latent KV” design; V3.2 then adds sparsity on top of that.

The key conceptual point you should internalize for DSA: **DeepSeek is not trying to make attention “smarter” by more compute; it’s trying to make attention “selective” by spending a tiny compute budget up front to avoid paying the quadratic bill everywhere.**


## Lesson 2 — Why “huge RL budgets” are hard, and what GRPO is buying you

Post-training with RL (RLHF-style or outcome-reward RL) is where modern frontier performance often gets “activated”: reasoning style, instruction-following stability, tool-use discipline, refusal behavior, etc. But scaling RL is notoriously brittle because it combines (i) nonstationary policies, (ii) sampling/truncation heuristics (top-p/top-k), (iii) distributed inference/training mismatches, and (iv) in MoE models, routing instability.

Your GRPO image is exactly the right place to anchor the intuition. DeepSeek-V3.2 says they adopt **GRPO** and then add a set of stabilizers that make “big RL compute” behave.

### GRPO in one mental model

Think: “PPO, but the baseline is _relative to a group of sampled responses_.” For each prompt q, sample a group `{o_1..o_G}` from the old policy, score each outcome with reward models or rule-based reward, then compute an advantage like:

`A_i = R_i - mean(R_over_group)` (you can literally see this in your screenshot)

That group-relative baseline reduces variance and makes optimization more about **ranking** within a local cohort than chasing an absolute reward scale. The objective then looks PPO-like: an importance ratio `r = pi_theta / pi_old`, a clipped surrogate loss, and a KL penalty to a reference policy.

### The DeepSeek V3.2 “RL scaling” crux: four stabilizers

DeepSeek then explicitly lists the “don’t let RL explode” mechanisms:

1. **Unbiased KL estimate.** They correct a KL estimator so the gradient isn’t systematically biased, which matters when sampled tokens become low-probability under the updated policy (classic source of noisy updates).
2. **Off-policy sequence masking.** They generate rollouts, then do multiple SGD steps; that introduces off-policy drift. They mask sequences (especially **negative-advantage** ones) that are too off-policy by a KL threshold, so bad, stale samples don’t destabilize training.
3. **Keep Routing (MoE-specific).** If routing differs between sampling and training, you’re optimizing different parameter subspaces—RL becomes chaos. They **freeze the routing path used at sampling and enforce it during training**.
4. **Keep Sampling Mask.** If you used top-p/top-k truncation during sampling, your training policy must respect the same action subspace or importance sampling assumptions break. They keep the truncation mask and apply it during training to preserve language consistency and stability.
    

This is why “huge RL budget” is not just throwing GPUs at PPO; it’s engineering the stochastic control loop so it doesn’t drift into garbage.

DeepSeek also states their framework allocates a **post-training compute budget exceeding 10% of pretraining cost**, which is a big deal because historically many open models spend far less there.


-----


## What DeepSeek-V3.2 actually changed (vs 3.1), and why it matters

### 1) The architectural delta: DSA is the only change from V3.1-Terminus

DeepSeek states plainly: compared to **DeepSeek-V3.1-Terminus**, the **only architectural modification** is introducing **DeepSeek Sparse Attention (DSA)** via continued training; V3.2 otherwise matches the V3.2-Exp architecture.

That matches your second/third images: the big backbone (MoE + MLA) stays; attention gets a new sparse path.

### 2) What DSA is: Lightning Indexer + fine-grained top-k token selection

DSA’s “prototype” is two components:

**Lightning Indexer.** For each query token, it computes an **index score** against each preceding token to estimate relevance. DeepSeek emphasizes that it’s efficient (few heads, FP8-friendly, ReLU for throughput).

**Fine-grained token selection.** Using those scores, DSA selects the **top-k KV entries** and runs attention only on that sparse set.

**This is exactly what third image shows: the green path is the indexer + top-k selector feeding a smaller KV set into core attention.**

A subtle but important point: DeepSeek instantiates DSA **under MLA** and implements it using **MQA mode** (KV entries shared across query heads) for kernel efficiency and to continue training from the existing checkpoint.  

**So it’s not really “MLA swapped with DSA”; it’s “MLA becomes the substrate, and DSA becomes the selector sitting above it.”**

### 3) The training trick that makes DSA work: “learn to imitate dense attention, then go sparse”

This is one of the most “first-principles clean” parts of the report.

They do continued pretraining in two stages:

**Dense warm-up stage (indexer alignment).** Keep dense attention, freeze everything except the indexer, and train the indexer to match the main attention distribution (aggregate attention across heads, normalize, minimize KL divergence). They train this warm-up briefly (they even give step/token counts).

**Sparse training stage (turn on selection).** Introduce top-k selection and train the whole model to adapt to sparse attention patterns, while still aligning indexer outputs to the attention distribution but only over the selected set. They also explicitly detach the indexer input for separate optimization: the indexer is trained by its KL alignment loss; the main model is trained by the LM loss.

They select **2048 KV tokens per query** (k=2048).

This “distill dense attention into a cheap scorer, then sparsify” recipe is the crux. It’s why the system doesn’t collapse quality when you stop attending to almost everything.

### 4) Complexity and cost: why this makes long context economically viable

DeepSeek states DSA reduces **core attention complexity** from quadratic to **O(L*k)** while the indexer still has an O(L^2) shape but much lower compute than MLA; combined with optimized implementation, they see significant end-to-end speedups in long context.

On the “cheaper than Gemini/GPT” claim: the paper itself focuses on _their_ inference cost curve on H800 clusters and shows the shape improvement with token position.  
For external price comparisons, DeepSeek’s own API announcement says V3.2-Exp prices dropped **50%+**. VentureBeat reports cached input at **$0.028 per 1M tokens** (and provides a comparison table across providers/models, with caveats).  
Whether it’s “10x cheaper than GPT/Gemini” depends on _which_ model tier, caching regime, and input/output mix you compare against; the defensible core is: **DSA flattens the long-context cost curve** and DeepSeek priced the API aggressively.

A systems-side implication you should care about: DSA complicates serving because you now have (i) indexer KV caches and (ii) sparse selection steps that interact with batching and paged attention. vLLM’s day-0 support writeup calls out these exact engineering issues (prefill vs decode handling, different cache layouts, top-k selection under batching).

---

## The other half of the story: specialists, distillation, mixed RL, and tool-use data at scale

### Specialist distillation and the “six specialists” you suspected

DeepSeek describes a post-training pipeline that includes **specialist distillation** and **mixed RL training**.  
They explicitly list six specialized domains: **mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, agentic search**, and they support both “thinking” and “non-thinking” modes.

The pipeline logic is: train specialists (with large-scale RL compute), use them to generate high-quality domain data, distill into the main model, then run mixed RL to remove remaining gaps and unify behaviors. DeepSeek claims the performance gap to specialists becomes marginal and can be eliminated through subsequent RL.

### Mixed RL training: unify reasoning + agents + alignment in one RL stage

They say they merge reasoning, agent training, and human alignment into a single RL stage (still GRPO-based) to balance performance and avoid catastrophic forgetting that often appears in multi-stage post-training.  
They also describe the reward mix: rule-based outcome reward, length penalty, language consistency reward for reasoning/agents, and a generative reward model with prompt-specific rubrics for general tasks.

### Tool-use and agentic synthesis: this is where the “agents create high-quality data” happens

DeepSeek describes a **cold-start** approach to combine reasoning traces with tool-call prompting, then scales up into large RL task sets.

They also make the “pipeline” claim concrete: they generate **over 1,800 environments** and **85,000 complex prompts** for agentic training data.  
They give an explicit task table with counts (for example: tens of thousands of code/search tasks; a few thousand general synthesized tasks), and distinguish real vs synthesized environments and extracted vs synthesized prompts.

The most “research-dev crux” details (the part you’d want in notes) are the pipeline patterns:

**Search agent:** sample long-tail entities, a question-construction agent explores via search tools with depth/breadth controls, multiple answer-generation agents create diverse candidates, then a verification agent validates and keeps only strongly-filtered samples; they also combine verifiable samples with rubric-scored helpfulness data via a generative reward model.

**Code agent:** mine issue/PR pairs, build _executable_ environments with an environment-setup agent that installs deps and runs tests; accept only when applying the gold patch fixes failures without regressions; they claim tens of thousands of such environments across many languages.

**General agent:** synthesize tool environments, then iteratively make tasks harder while keeping verification easy; retain only tasks with non-zero pass@100 after RL.

This is why DeepSeek’s story isn’t just “new attention kernel.” It’s a coordinated package: **efficient long-context attention + RL scaling stability + industrial-scale synthetic/real tool environments + specialist-to-general distillation.**

---

## “Limit testing” and practical constraints (what breaks in real agent runs)

DeepSeek notes that in tool benchmarks the model sometimes produces redundant self-verification and overly long trajectories that can exceed the 128K context limit; they use context management for scoring and identify this as a practical direction for improvement.  
They also describe a specific “thinking context management” rule for tool-calling: keep reasoning traces across tool outputs and only discard when a new user message arrives, to avoid re-reasoning from scratch every tool call.

---

## The landscape impact (how to file this in your mental map)

DeepSeek-V3.2 is best seen as a concrete instance of a broader trend: **stacking sparsity and stacking training signals**.

1. **Compute sparsity in the FFN** (MoE): only some experts fire per token (your second image).
    
2. **Attention sparsity in long context** (DSA): only k past tokens participate in attention per query, chosen dynamically by a learned scorer (your third image).
    
3. **Post-training compute scaling** (GRPO + stabilizers): treat RL as a major budget item, but engineer it so it remains stable at scale (your first image).
    
4. **Data synthesis as infrastructure**: build agentic environments and verifiable tasks so RL has a wide, hard curriculum rather than fragile hand-designed prompts.
    

If you connect this back to your earlier Engram interest: Engram was “conditional memory lookup” (static knowledge via O(1) access). DSA is “conditional context lookup” (dynamic sparse access to the relevant slice of the prompt). Different layers of the same idea: **stop paying full neural compute for information that can be cheaply selected or directly indexed.**

---

## If you’re turning this into notes, capture these three invariants

(1) DSA is not magic; it’s **learned imitation of dense attention distributions**, followed by sparse execution with k=2048.  
(2) “Huge RL compute” only works if you control off-policy drift, routing drift, and sampling drift; DeepSeek names the concrete mechanisms (unbiased KL, sequence masking, keep routing, keep sampling mask).  
(3) The agentic jump is mostly a **data/verification pipeline story** (real tools + synthetic environments + multi-agent generation + verification), not only a model-architecture story