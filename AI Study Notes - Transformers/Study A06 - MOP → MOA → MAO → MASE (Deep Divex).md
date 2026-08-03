

## 1. Core mental model

The four “paradigms” are best understood as four different _control loops_ by which a language model’s behavior gets shaped. Each loop differs along three axes: what gets updated (weights vs prompts/memory/tools vs policies over actions), where the data comes from (static corpus vs curated supervision vs interaction transcripts vs environment outcomes), and what the optimization signal is (token log-likelihood vs labeled loss vs preference/reward vs task success metrics). The same base transformer can sit inside any of these loops; the “learning paradigm” is primarily about the _outer loop_ you wrap around it.

A clean way to think about the sequence MOP → MOA → MAO → MASE is: you start by learning a broad conditional distribution over text (MOP). Then you constrain that distribution to be useful and aligned for human intents (MOA). Then you stop treating “one model call” as the unit of intelligence and instead learn/engineer _interactions among multiple calls, roles, tools_ (MAO). Finally you let the system update itself based on outcomes in an environment, so improvement is continuous and not confined to an offline training job (MASE). The crucial shift is that “learning” moves from parameter space to system space: first weights, then behavior shaping, then orchestration policies, then self-improving closed loops.

Each loop differs by (i) **what is being optimized/updated**, (ii) **where the data/feedback comes from**, and (iii) **what the objective signal is**. The transformer is just a powerful conditional density model inside the loop; the “learning paradigm” is the **outer loop design**.

---
There are two separations that keep thinking crisp:

**(A) Parameter learning vs system learning.**  
MOP and most of MOA primarily update _model parameters_ (or PEFT deltas). MAO and MASE often leave the base model fixed and instead update the _system that wraps the model_: prompts, role routing, tool policies, memory schemas, verification gates, retrieval, evaluation harnesses, and decision thresholds.

**(B) Static-data optimization vs interaction optimization.**  
MOP is fit to a static distribution of text. MOA introduces curated supervision and preference signals (often still offline). MAO and MASE operate on **trajectories**: multi-step interactions, tool calls, and environment outcomes.

A unifying mental picture: the effective mapping from an input question to an answer is not “a model call,” it is a computation graph that may include retrieval, planning, tool invocation, verification, critique, and synthesis. Over time, you can improve that computation graph (MAO/MASE) even if the base model stays fixed, and that often buys you more reliability per unit cost than another round of fine-tuning.

---

## 2) Key objects and math skeleton 

### 2.1 MOP (Model Offline Pretraining): learn `pθ(x_t | x_<t)` by compression

Let a token sequence be `x = (x1,…,xT)` and the transformer define:

`pθ(x) = ∏_{t=1..T} pθ(x_t | x_<t)`

Training is maximum likelihood (cross entropy) on a fixed corpus `D`:

`L_pre(θ) = E_{x~D} [ Σ_t -log pθ(x_t | x_<t) ]`

Interpretation: the model becomes a strong **conditional compressor**. Any structure that helps predict text (syntax, facts, reasoning-like patterns, code idioms) is incentivized because it reduces average negative log-likelihood. “Capabilities” are not inserted explicitly; they emerge if they are predictive regularities in `D`.

### 2.2 MOA (Model Online Adaptation): behavior shaping via supervision + preferences

**Supervised fine-tuning (SFT)**: instruction/context `x`, target response `y`.

`L_sft(θ) = E_{(x,y)~S} [ -log pθ(y | x) ]`

SFT reduces entropy around “desired” completions: among many plausible continuations, it puts probability mass on the style/format/intent you want.

**Preference alignment** introduces comparisons. Data: `(x, y⁺, y⁻)` where `y⁺` is preferred.

Two common skeletons:

Reward modeling: learn `rφ(x,y)` s.t. `rφ(x,y⁺) > rφ(x,y⁻)`; then optimize policy `πθ` with a trust-region to a reference `π_ref`:

`maximize_θ E_{y~πθ(.|x)} [ rφ(x,y) ] - β * KL(πθ(.|x) || π_ref(.|x))`

DPO-style direct preference: skip explicit RL; change likelihood ratios so preferred samples become more likely than dispreferred, still anchored to a reference model. The “shape” is: **increase log-prob of `y⁺` relative to `y⁻`**, with strength controlled by a temperature/scale and proximity to `π_ref`.

**PEFT (LoRA/adapters)** changes what is updated. Instead of changing full `θ`, you learn a small delta `Δ`:

`θ := θ0 + Δ` where `Δ` is low-rank or small modules.

This stabilizes training, reduces storage, and makes it easier to maintain many task/persona variants.

### 2.3 MAO (Multi-Agent Orchestration): optimize a policy over trajectories

Now your “system” produces a trajectory `τ` of messages and tool calls.

State at step `t`: `s_t = {history, retrieved context, tool outputs, agent roles, constraints, budget}`  
Action: `a_t = {which agent acts, what prompt/template, message/tool call}`

Objective is expected utility over trajectories:

`J(π_orch) = E_{τ~π_orch} [ U(τ) ]`

`U(τ)` is task-dependent: correctness score, judge rating, safety constraints, cost/latency penalties, etc. Notice: you can keep the base LLM fixed; learning/improvement can happen in `π_orch` (routing), prompt libraries, role decomposition, tool schemas, and verification.

### 2.4 MASE (Multi-Agent Self-Evolving): online learning with environment feedback

Add an environment `E` that returns observations and outcomes:

`s_{t+1}, r_{t+1} = E(s_t, a_t)`

Now you can do continual improvement: update orchestration policy, prompts, memory, retrieval parameters, and sometimes model weights—based on logged outcomes. Depending on whether actions affect future states, this resembles contextual bandits (myopic) or RL (long-horizon).

A critical systems principle: separate **fast-changing** components (prompts, routing thresholds, retrieval parameters) from **slow-changing** components (model weights), and gate slow updates with strong eval to prevent drift.

---
---

## 3) Algorithmic / process flow (mental simulation of each loop)

### 3.1 MOP training loop (SGD on tokens; no interaction)

Static corpus D
   -> tokenize batches
      -> forward pass (Transformer)
         -> logits over vocab
            -> cross-entropy loss
               -> backprop gradients
                  -> update θ


Key iteration unit: “predict next token.”  
The model never sees “success/failure” on tasks; it sees only prediction error.

### 3.2 MOA loop (SFT + alignment, often staged)

A common production pipeline:

**Stage A: SFT**

Instruction pairs S: (x,y)
   -> forward
      -> -log pθ(y|x)
         -> update (θ or PEFT Δ)


**Stage B: preference alignment**

Preference data P: (x, y+, y-)
   -> either train reward model rφ OR apply direct preference objective
      -> update policy πθ with proximity constraint to π_ref


**Stage C: deployment controls**  
System prompts, refusal policies, safety classifiers, and evaluation harnesses.

What changes qualitatively: SFT shapes “how to respond.” Preference alignment shapes “which of many plausible responses is preferred,” and the KL/anchoring term is what prevents brittle over-optimization.

### 3.3 MAO loop (runtime orchestration; learning is in structure)

User task -> build shared state s0
   -> ROUTE: pick role/agent (planner/solver/critic/verifier/retriever)
      -> ACT: generate message OR call tool
         -> OBSERVE: tool output / new evidence
            -> CHECKS: constraints, consistency, citation, unit tests, guardrails
               -> iterate until "done" or budget exhausted

Important: “multi-agent” is not magic; it is **structured decomposition** plus **cross-checking**. You’re constructing a computation graph that makes failures less likely and makes the system grounded via tools.

### 3.4 MASE loop (MAO + telemetry + self-updates)

Add two blocks: **Telemetry** and **Update**.

Telemetry:

log: prompts, routes, tool outputs, final answer, cost/latency
log: eval scores, user edits, regressions, safety flags


Update:

periodically:
   -> analyze failures / win cases
   -> adjust: routing policy, prompt library, memory schema, retrieval hyperparams
   -> curate new training traces (optional)
   -> run offline training + eval gates (rare for weight updates)
   -> roll forward or rollback


This is where “agentic systems improve”: not by mystical self-awareness, but by **closed-loop optimization + disciplined evaluation**.


---
---


## 4) Why it behaves the way it does (lenses that matter here)

### 4.1 Function-space lens: what’s being learned is a mapping class

MOP learns a broad conditional function `fθ: context -> distribution over tokens`. Because the loss averages over an enormous corpus, `fθ` approximates a “universal prior” over text-like continuations. It will represent latent features that are useful across many contexts because those reduce prediction error globally.

MOA is best seen as **function editing / re-weighting**: you are not typically adding new primitives from scratch; you are pushing probability mass toward behaviors that match your supervised or preference distribution. PEFT makes this explicit: `fθ0` stays, and `Δ` nudges outputs in targeted regions.

MAO and MASE shift the effective function from `fθ` to a composed operator:

`F = Compose( Retrieve, Plan, ToolCall, Verify, Critique, Answer )`

Even if `fθ` is unchanged, `F` can be dramatically stronger because it has external state, multi-step computation, and grounding. This is why orchestration can outperform fine-tuning for reliability: you’re changing the _inference computation graph_, not just the weights.

### 4.2 Loss landscape / stability lens: why anchoring terms exist

MOP’s objective is smooth and data-rich; gradient noise is high-dimensional but averaged over huge samples. MOA’s preference signals are more sparse and can be “sharp” (small dataset, high leverage). This increases risk of moving into brittle regions (mode collapse, overfitting, reward hacking). The proximity term:

`β * KL(πθ || π_ref)`

is not decorative—it is a stability control. It keeps the policy near a known-good distribution and prevents exploiting artifacts in the reward/preference model.

### 4.3 Bias–variance and distribution shift lens

MOP has broad coverage but inherits dataset bias (what the web emphasizes). MOA reduces bias for your target tasks (instruction-following, safety) but can increase variance if the adaptation set is narrow, leading to overspecialization or “forgetting.”

MAO reduces variance via redundancy: multiple roles, verification, tool grounding. But it introduces new systemic error modes: routing mistakes, correlated agent failures (same model style), tool mis-specification, and compounding latency/cost.

MASE is where distribution shift is explicitly handled, but stability becomes the main problem: online feedback is noisy and adversarially exploitable; without drift detection, evaluation gating, and rollback, the system can self-degrade.

---

## 5) Relations and contrasts (how to choose and combine)

**MOP vs MOA**: choose MOP/continued pretraining when you need broad domain competence or new knowledge priors at scale (e.g., a new programming language corpus, specialized scientific literature). Choose MOA when the model “knows enough” but behaves incorrectly: poor instruction following, wrong format, unsafe defaults, mismatch with user intent, insufficient preference alignment.

**MOA vs MAO**: MOA changes the distribution the model samples from; MAO changes the _protocol_ by which you obtain an answer. If your failures are about factuality, tool grounding, multi-step tasks, and reliability under constraints, MAO is often higher ROI than more fine-tuning because it introduces explicit checks and external computation.

**MAO vs MASE**: MAO is engineered orchestration; MASE adds continual improvement. If your environment is stable, MAO plus strong eval and prompt/tool iteration may be enough. If your environment shifts (tools change, data evolves, user intents drift) and you need persistent learning, MASE is the right frame—but you must treat it like an online learning system: define metrics, guardrails, golden test suites, and rollback.

A practical “stacked” view (common in real systems):

Base model from MOP
   -> MOA (SFT + preference alignment, often PEFT)
      -> MAO (retrieval + tools + verifier + multi-role loops)
         -> MASE (telemetry + auto-iteration + periodic retraining with eval gates)


---

### Minimal ASCII “four loops” summary diagram

(1) MOP:  static text ------------------> θ  (maximize likelihood)

(2) MOA:  (x,y) supervision + (x,y+,y-) prefs --> θ or Δ  (behavior shaping)

(3) MAO:  orchestrator π_orch over multi-step trajectories τ
          agents + tools + verifiers ----------> better U(τ) (no weight change needed)

(4) MASE: MAO + environment feedback + online updates
          telemetry -> update prompts/routing/memory/tools/(rarely θ) -> improved U over time
