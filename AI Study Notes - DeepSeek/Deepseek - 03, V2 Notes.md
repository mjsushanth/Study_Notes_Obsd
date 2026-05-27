
From HCI Class, long ago. “DeepSeek V3 + R1/R1-Zero deep-dive notes,” 

---

## 1. Core mental model: what “DeepSeek” actually is

At a high level, **DeepSeek is two big ideas glued together**:

1. **DeepSeek-V3**: a *hardware-conscious, sparse Mixture-of-Experts (MoE) transformer*
   – 671B total parameters, but only ~37B active per token, trained on ~14.8T tokens.
   – Uses a custom MoE stack (“DeepSeekMoE”), Multi-Head Latent Attention (MLA), multi-token prediction, FP8 training, and no auxiliary load-balancing loss. ([Hugging Face][1])

2. **DeepSeek-R1 / R1-Zero**: a *reinforcement-learning-driven reasoning layer* on top of a strong base model
   – R1-Zero: pure RL on a base model without SFT or a learned reward model, using a GRPO-style algorithm. ([OpenReview][2])
   – R1: a multi-stage pipeline (cold-start SFT → large-scale RL → rejection sampling → RL again) that shapes the model into a general reasoning assistant. ([arXiv][3])

So the mental picture is:

> **V3** = “efficient, sparse, high-capacity function class”
> **R1-Zero/R1** = “how we sculpt that function class into a step-by-step reasoner using RL, not just SFT”

Your lecture response is essentially walking from “LLMs and reasoning” → “MoE architectures” → “R1-Zero as a pure RL experiment” → “R1 as the refined training recipe.” I’ll mirror that trajectory.

---

## 2. DeepSeek-V3 as a function class: MoE, MLA, multi-token prediction

### 2.1 What V3 is solving

Dense transformers hit a wall:

* Scaling laws suggest you want more parameters and more tokens, but **dense compute** becomes painful.
* You want specialization: different parts of the network should focus on math, code, long-context reasoning, etc., without paying full compute everywhere.
* You also want to reduce **test-time compute** for a given quality level.

DeepSeek-V3’s answer is a **sparse MoE transformer**:

* Keep a very large total parameter budget (many experts).
* For each token, **activate only a small subset of experts**, so compute per token ~ 37B params instead of 671B. ([Hugging Face][1])

### 2.2 Key architectural objects (math skeleton)

At a single MoE layer, ignoring attention for a moment:

* Input token hidden state: `h ∈ R^d`
* There are `E` experts: each expert is a feed-forward block `f_k(h; θ_k)`
* A router/gating network produces scores over experts: `g(h) ∈ R^E`
* We select top-K experts per token and mix their outputs.

Formally, for one token:

* Router logits: `z = W_r h + b_r`
* Gating probabilities: `p_k(h) = softmax(z)_k`
* Top-K experts chosen: `S(h) ⊂ {1…E}`, |S(h)| = K
* Expert outputs: `y_k = f_k(h)` for k in S(h)
* MoE output:
  `MoE(h) = Σ_{k ∈ S(h)} α_k(h) · y_k`

where `α_k(h)` are normalized gating weights (maybe rescaled). The implementation has a lot of engineering (capacity constraints, routing strategies) but structurally it’s:

> “Small router decides which 2–4 experts to apply; you only run those networks for this token.”

DeepSeek-V3 innovates on several axes: ([arXiv][4])

* **DeepSeekMoE**: hierarchical / structured experts, and an **auxiliary-loss-free load balancing** strategy so you don’t need a separate balancing loss term (unlike Switch Transformer / GShard).
* **Multi-Head Latent Attention (MLA)**: attention redesigned to be more memory-efficient by sharing projections and using latent spaces, reducing KV cache size and improving long-context behavior.
* **Multi-token prediction**: training objective predicts several future tokens at once, improving sample efficiency.
* **FP8 mixed precision**: lower precision activations/weights for cheaper training and inference.

From a function-space view, you can think of this as:

* A gigantic collection of specialized nonlinear functions `{f_k}`,
* A small, learned “router” that, for each token, chooses a tiny subset of these functions to combine.

### 2.3 Algorithmic flow of a V3 layer (forward pass)

For one transformer block in V3:

1. **Attention sublayer**
   * Possibly MLA: project `h` into a latent space, compute Q/K/V with more compact representations, attend, and project back.
   * Output `h_attn`.

1. **MoE feed-forward sublayer**
   * Compute router logits and gating `p(h_attn)`.
   * Select top-K experts per token subject to capacity (each expert gets only so many tokens per batch).
   * Compute `y_k = expert_k(h_attn)` for selected experts.
   * Weighted sum to get `h_ffn = Σ α_k y_k`.

1. **Residual / normalization**
   * `h_out = h_in + Dropout(h_attn + h_ffn)` with appropriate RMSNorm/LayerNorm flavors.

During training, the token-expert assignment plus the multi-token prediction objective are optimized jointly. The auxiliary-loss-free router design means the load balancing emerges from routing constraints and gating design instead of an explicit extra term. ([arXiv][4])

### 2.4 Why this architecture works (qualitative behavior)

* **Parameter efficiency**: You get the *representational richness* of 671B parameters, but **compute cost** closer to a ~30–40B dense model at inference.
* **Soft modularity**: Experts can specialize (code, math, NL, multilingual nuance) while the router learns when to activate them.
* **Compute focusing**: For harder tokens (complex math step, nontrivial reasoning), the router can activate more suitable experts and allocate higher capacity implicitly.

Empirically, DeepSeek-V3 reaches GPT-4-ish performance on math/physics/coding, competing with much larger dense models at lower compute cost. ([Medium][5])

From lecture notes: V3 is the **strong base** you need before reasoning-focused RL can shine. If the base model has weak representations, RL on top is just flailing noise.

---

## 3. From base LLMs to reasoning: chain-of-thought and test-time compute

Before we touch R1-Zero, it’s useful to align with the lecture context you summarized:

* **Base LLMs** (even very big ones) are essentially **pattern completion engines**: given text, they predict the next token. They don’t naturally “follow instructions” or “show their work.”
* **Instruction tuning (SFT)** takes that base and teaches “how to respond to a user”: structure, style, roles, etc.
* **Chain-of-thought (CoT)** training adds supervision on the *reasoning path*: the model outputs step-by-step thinking before the final answer.

Scaling laws and **test-time compute** then become key:

* Performance improves with deeper “thinking” (more tokens and more operations per query), but that costs compute at inference.
* There is a trade-off: model size vs. reasoning depth vs. inference budget.

DeepSeek’s innovation is to say:

> Instead of only improving the *prior* (architecture + pretraining), let’s explicitly train the model via **reinforcement learning** so that, for reasoning tasks, it **learns when and how to think longer**.

That’s where **R1-Zero** enters.

---

## 4. R1-Zero: pure RL on a base model (no SFT, no reward model)

### 4.1 Conceptual leap

Classical RLHF pipeline:

* Base model → SFT (human or synthetic instruction data) → reward model (trained on pairwise preferences) → PPO/variants for RL.

R1-Zero does something much more aggressive:

* Take a base math / reasoning model.
* **No SFT**, no learned reward model.
* Apply RL *directly* on a relatively small set of rule-checkable problems (~8k math problems in replication studies), with handcrafted rewards. ([arXiv][6])

The reward is computed from:

* Correctness of final answer.
* Structural tags (`<think>…</think>`, `<answer>…</answer>`).
* Sometimes formatting / brevity constraints.

The idea: if you can define the reward algorithmically (e.g., check math answer), you don’t need humans or a reward model.

### 4.2 GRPO: Group Relative Policy Optimization (math skeleton)

R1-Zero uses a variant of policy gradient RL called **GRPO** (Group Relative Policy Optimization). ([OpenReview][2])

Roughly:

* For each prompt `x`, the current policy `π_θ` generates a *group* of candidate completions `{y₁, …, y_K}`.
* Each candidate gets a scalar reward `r_i` from rule-based checks.
* Instead of training a separate value function, you compute a *baseline* from the group itself (e.g., mean reward `\bar r`).
* Advantage: `A_i = r_i − \bar r`.

Policy gradient estimate:

* `∇_θ J(θ) ≈ E[ A_i ∇_θ log π_θ(y_i | x) ]`

Then you add constraints like KL penalties to keep updates from drifting too far from the base model distribution (PPO-style clipping / trust region).

In words:

> The model is rewarded if it does *better than its own group average* on this prompt; punished if worse. No explicit value model.

### 4.3 Algorithmic flow of R1-Zero training

For each RL step:

1. **Sample a batch of prompts** (e.g., math problems).
2. **Generate multiple candidate solutions per prompt** with current model, using the `<think>` / `<answer>` format.
3. **Evaluate each candidate** with rule-based functions:

   * Is the final answer correct?
   * Are tags well-formed?
   * Penalize pathological outputs (no answer, wrong format, etc.).
4. Within each prompt’s group, compute **relative rewards** and baselines.
5. Compute **policy gradients** using GRPO (group-relative advantage) with trust-region / KL constraints.
6. **Update model parameters**.
7. Repeat for many steps; monitor metrics like pass@k on math benchmarks, CoT length, etc.

Empirically, as you noted in your response:

* **Chain-of-thought length grows** with training: the model chooses to “think longer” on harder problems.
* You see an **“aha moment”**: at some training stage, performance and reasoning structure jump—self-reflection, error checks, course corrections start to appear. ([arXiv][3])

This is why people talk about “emergent meta-learning”: the policy learns *how to allocate its own test-time compute* and how to structure reasoning traces, purely driven by reward.

---

## 5. DeepSeek-R1: full reasoning pipeline on top of V3

R1-Zero is the pure experiment; **R1** is the actual production-grade recipe.

DeepSeek-R1 typically follows a **multi-stage pipeline** roughly like: ([arXiv][3])

1. **Strong base model**

   * Start from DeepSeek-V3-Base: huge MoE model trained on 14.8T tokens.

2. **R1-Zero-style RL cold-start**

   * Apply R1-Zero-like RL on math/reasoning problems to get a reasoning-focused checkpoint (strong at CoT and correctness, but narrow domain).
   * This may use only ~8k–tens of thousands of math / reasoning problems with automatic rewards.

3. **Synthetic SFT (“cold-start SFT”)**

   * Use the RL-enhanced model to generate high-quality reasoning traces for many more synthetic or curated tasks.
   * Build an SFT dataset of (prompt, chain-of-thought, answer) triples.
   * Fine-tune on this to stabilize behavior and widen the domain slightly.

4. **Large-scale RL for reasoning**

   * Run a bigger RL stage on a wide set of reasoning problems, again using GRPO-like algorithms and structured CoT output.
   * This is where the model’s reasoning ability is pushed to near-frontier levels.

5. **Rejection sampling + general SFT**

   * To turn a math-centric reasoner into a usable assistant, they:

     * Sample responses from the RL checkpoint on a mixture of reasoning and general prompts.
     * Apply *self-ranking / rejection sampling* to keep the best candidates.
     * Combine those with existing supervised data (writing, factual QA, self-cognition, safety) to train a new SFT model. ([arXiv][3])

6. **Final RL polishing**

   * Run another RL stage on this broader model to regain some of the reasoning sharpness lost during general SFT.
   * Objective: a **general assistant** that still retains strong CoT reasoning.

The cost discussion:

* Reports quote roughly **single-digit millions of dollars** in GPU cost to train V3, not counting R&D and R1 experiments. ([Interconnects][7])
* Your lecture response noted that the famous “$5M” figure usually omits R&D and pretraining for R1; it’s best read as a *hardware cost for one successful run*, not the total real-world cost of the whole program.

---

## 6. Distillation and replication: R1’s effect on the broader ecosystem

One of the striking things in your notes and in the literature is **how replicable the R1-Zero trick turned out to be**:

* Independent groups took small base models (Llama3, Mistral, Qwen, DeepSeekMath) and applied a very simple RL recipe on ∼8k GSM8K/Math problems.
* They got **10–20 point improvements** on reasoning benchmarks with just rule-based rewards and RL; no reward model, no massive SFT. ([GitHub][8])

This led to:

* A wave of **distilled “R1-like” models**, where small LLMs are trained to mimic R1’s chain-of-thought outputs (open-sourced in many repos).
* A growing belief that we may have **overcomplicated RLHF** (full preference models, huge SFT phases) when simple GRPO-style RL on good base models and small rewardable datasets already yields large gains.

Conceptually, the pattern is:

> Strong base model + small but high-signal RL loop + self-distillation → a family of smaller, cheaper reasoning models.

That’s the “democratization” story you see in commentary: you don’t need OpenAI-scale budgets to get “good enough” reasoning anymore. ([Leanware][9])

---

## 7. How MoE + RL interact (what’s special about DeepSeek’s combo)

Now, connecting the two layers:

1. **MoE gives you cheap conditional capacity.**

   * You can afford 671B total parameters because only ~37B are active per token.
   * This matters when RL makes the model **think longer** (more tokens in `<think>`): the active compute per token is lower, so long reasoning traces are less painful.

2. **RL shapes *which experts fire* and *how long the model thinks*.**

   * In R1-like training, the router and experts collectively learn to support long-horizon reasoning with reflection, backtracking, etc.
   * Intuitively, some experts may specialize in symbolic manipulation, others in textual explanation or meta-cognition.

3. **Test-time compute as a learned decision.**

   * R1-Zero / R1 show that as RL progresses, the model learns when to extend the `<think>` section; hard problems yield longer chains. ([arXiv][3])
   * On an MoE backbone, those extra tokens are processed sparsely, which keeps inference cost manageable relative to performance gains.

Your lecture’s discussion questions about:

* “High-reward expert bias”
* “Expert pruning effects”
* “Standard attention vs latent attention”
* “Memory retention across long CoT without full quadratic attention”

are basically digging into how **routing + RL co-adapt**: which experts become dominant under RL’s reward structure, whether some experts become dead or pruned, and how MLA + MoE handle long reasoning histories without blowing up KV cache and attention cost.

---

## 8. Failure modes and open research questions

Some of the issues hinted in your notes and current papers:

1. **Training instability in RL for LLMs**

   * Large policy updates can break the model (collapse, weird modes).
   * GRPO avoids a value network but still needs careful KL constraints and learning rate tuning to avoid “policy drift.” ([OpenReview][2])

2. **Reward hacking and reasoning shortcuts**

   * If rewards are too simplistic (e.g., only final answer correctness), the model may find brittle strategies or exploit spurious patterns.
   * “High-reward expert bias” can cause the router to overuse some experts that happen to align with reward quirks, reducing diversity.

3. **Exploration vs scaling**

   * MCTS-style exploration on CoT trees is expensive; naive search explodes combinatorially.
   * R1-Zero style relies on random sampling and learning from those; there’s an open question whether more structured exploration can be made tractable.

4. **Generalization beyond math / synthetic tasks**

   * The math domain is nice because reward is easy to compute.
   * For fuzzier domains (ethics, open-ended reasoning), the same RL trick requires new reward designs (learned or rule-based) and has more failure modes.

5. **Interpretability of expert activation patterns**

   * How exactly do expert usage patterns differ between math vs. code vs. natural language reasoning tasks?
   * Can we identify “reasoning experts” vs. “memorization experts”?
   * What happens under pruning or sparsity constraints?

These are the kind of PhD-scale questions your response was already hinting at.

---

## 9. Relations and contrasts

A few useful comparisons to keep in your head:

1. **DeepSeek-V3 vs dense LLMs (GPT-4, Llama-3, etc.)**

   * Dense models: same parameters active on every token; simpler conceptually, heavier per-token compute.
   * V3: sparse MoE; more parameters in total, but cheaper per token; more complex routing + balancing issues.

2. **R1-Zero/R1 vs classical RLHF**

   * Classical RLHF: SFT → reward model from human preferences → PPO-like RL.
   * R1-Zero: no SFT, no reward model, pure RL on auto-checkable tasks (math) with GRPO.
   * R1: minimal SFT, heavy RL, self-generated SFT via rejection sampling; much more RL-centric than human-label-centric.

3. **R1 approach vs OpenAI o1**

   * Both emphasize step-by-step reasoning and long CoT with RL.
   * DeepSeek’s recipe is largely public and reproducible; several papers explicitly analyze “R1-like training.” ([OpenReview][2])
   * Governance, data provenance, and safety regimes differ, but methodologically the big idea—**use RL to incentivize thinking**—is shared.

---

## 10. Cheat-sheet / interview-style recap

If you had to compress all of this into something you’d say in an interview:

* **What is DeepSeek-V3?**
  A very large sparse Mixture-of-Experts transformer (671B total, ∼37B active per token) trained on ~14.8T tokens. It uses DeepSeekMoE, multi-head latent attention, FP8, and multi-token prediction to get GPT-4-level performance with much lower per-token compute. ([arXiv][4])

* **What is DeepSeek-R1-Zero?**
  A pure RL experiment that applies Group Relative Policy Optimization directly to a base LLM on a small set of auto-gradable math problems—no SFT, no reward model. It learns to produce long chain-of-thought traces and shows emergent self-reflection as a function of RL training. ([OpenReview][2])

* **What is DeepSeek-R1?**
  A full pipeline that starts from V3-Base and goes through: small cold-start SFT, large-scale RL for reasoning, rejection sampling to form a new SFT set, then another RL round. The result is a general assistant with very strong reasoning, whose training recipe is fairly transparent and replicable. ([arXiv][3])

* **Why is this important?**
  It shows:

  * Sparse MoE can give you frontier performance at much lower per-token compute.
  * RL with simple rule-based rewards on small datasets can dramatically boost reasoning, and these gains can be distilled into smaller models.
  * A lot of the “magic” is: **strong base model + RL that rewards explicit thinking + self-distillation**, not just more SFT.

* **Core mental model to remember:**

  > DeepSeek is an example of “**architecture for efficient capacity (V3)** + **RL for emergent reasoning (R1-Zero/R1)** + **distillation for democratization**.”
  > The MoE backbone gives cheap conditional compute; the RL recipe tells the model when/how to think; distillation spreads that ability to smaller checkpoints.

---


[1]: https://huggingface.co/deepseek-ai/DeepSeek-V3?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V3"
[2]: https://openreview.net/pdf?id=jLpC1zavzn&utm_source=chatgpt.com "Understanding R1-Zero-Like Training: A Critical Perspective"
[3]: https://arxiv.org/pdf/2501.12948?utm_source=chatgpt.com "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs ..."
[4]: https://arxiv.org/html/2412.19437v1?utm_source=chatgpt.com "DeepSeek-V3 Technical Report"
[5]: https://medium.com/%40sanatsharma/deepseek-v3-technical-deep-dive-cf8ea5e7b78a?utm_source=chatgpt.com "DeepSeek-V3 — How to Make a Performant Language ..."
[6]: https://arxiv.org/html/2502.02523v1?utm_source=chatgpt.com "Brief analysis of DeepSeek R1 and it's implications for ..."
[7]: https://www.interconnects.ai/p/deepseek-v3-and-the-actual-cost-of?utm_source=chatgpt.com "DeepSeek V3 and the cost of frontier AI models"
[8]: https://github.com/hkust-nlp/simpleRL-reason?utm_source=chatgpt.com "hkust-nlp/simpleRL-reason: Simple RL training for reasoning"
[9]: https://www.leanware.co/insights/what-is-deepseek?utm_source=chatgpt.com "What Is DeepSeek? Everything You Need to Know in 2025"
