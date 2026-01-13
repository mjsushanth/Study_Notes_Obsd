

Short, high-signal bullets, follow-ups & concise answers, plus 90–120s spoken scripts you can rehearse. 

1. Elevator / Cheat-sheet (one-page, scanable)
2. Deep-intuition bullets per topic (the “explain like I’m on an interview panel” version)
3. Common followups + short answers (ready for quick fire questions)
4. Ten 90–120s spoken scripts you can rehearse (one for each core topic)

---

# 1) Elevator / Cheat-sheet (scan fast — memorize these)

* Encoder-Decoder: encoder → per-token representations $H=[h_1..h_L]$. Decoder is autoregressive: produces $P(y_t\mid y_{<t},H)$ for each output position.
* Softmax & CE: $p_i=\exp(z_i)/\sum_j\exp(z_j)$. CE/NLL for true class $t$: $\mathcal{L}=-\log p_t$. Gradient: $\partial\mathcal{L}/\partial z = p-y$.
* Scaled Dot-Product Attention: $S=QK^\top/\sqrt{d_k}$, $A=\text{softmax}(S)$, $Y=A V$.
* Multi-Head: independent linear proj per head, attend in parallel, concat heads, project: $ \text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_i)W^O$.
* KV caching (decoding): cache per-layer per-head $K,V$; reuse for each new token → amortized compute, lower latency.
* Backprop through attention (sketch): $dV=A^\top dY,\; dA=dY V^\top,\; dS=\text{softmaxJac}(dA,A),\; dQ=dS\;K/\sqrt{d},\; dK=(dS)^\top Q/\sqrt{d}.$
* Optimizers: Adam = momentum $m$ + RMS $v$ + bias-correction; AdamW decouples weight decay.
* MoE: many experts, sparse gating (top-k), conditional compute → massive capacity with low per-token FLOPs; engineering hard (routing, load balance, all-to-all).
* ViT vs conv-stem: ViT patches = global attention on tokens; conv-stem = local feature extractor → helps small datasets by building local inductive bias.

---

# 2) Deep-intuition bullets (interview ready)

Below each topic: 2–4 crisp bullets (what to say + why it matters + one short formula where relevant).

## RNN / LSTM fundamentals

* What: RNNs compute $h_t = f(x_t, h_{t-1})$; LSTM adds gates (input/forget/output) to control information flow and mitigate vanishing gradients.
* Why: gating lets LSTM keep long-term memory (cell state) while selectively writing/reading — so it learns longer dependencies than vanilla RNN.
* Interview lines: “Same weights across time (parameter sharing) — different inputs produce different hidden outputs $h_t$, a per-time compact state.”
* Practical: use layer norm, gradient clipping; LSTMs still useful for time series and where recurrence is natural.

## Seq2Seq (RNN era) + attention

* What: encoder compresses source into states $h_1..h_L$; decoder uses last state (or attention) to generate output tokens.
* Why attention: avoids single-vector bottleneck; compute scalar alignment $e_{t,i}=f(s_{t-1},h_i)$, normalize $\alpha_{t,i}=\text{softmax}(e_{t,i})$, context $c_t=\sum_i \alpha_{t,i}h_i$.
* Interview line: “Attention gives decoder direct dynamic access to encoder positions, so gradients/information don’t have to travel through one vector.”

## Transformer core (attention → no recurrence)

* What: replace recurrence with self-attention blocks + FFN; stacked layers + residuals + LayerNorm. Decoder: masked self-attn → cross-attn (queries from decoder, keys+values from encoder) → FFN.
* Why: parallel training, global receptive field, better scaling.
* Implementation note: Q/K/V are linear projections of inputs; softmax along keys (axis = sequence dimension) yields attention weights per query.

## Keys, Queries, Values (one clear mental model)

* Q = “what I’m looking for”; K = “what each memory slot offers (index)”; V = “payload to read if key matches”.
* Math: $E=QK^\top/\sqrt{d}$, $A=\text{softmax}(E)$ (softmax over keys), $Y=A V$.
* Why normalize over keys (dim=1 / axis of K): each query distributes its mass across keys (choose which memory slots to read).

## Scaled dot-product & Softmax intuition

* Dot product measures similarity (alignment between Q and K). Scaling by $\sqrt{d}$ keeps variance ≈ 1 so softmax gradients aren’t too small.
* Softmax over keys: each query decides how much attention to give to each key (probabilities sum to 1 per query).

## Multi-head self-attention (MHSA)

* What: run several small-dimensional attention heads in parallel: heads capture different relations (syntax, locality, coref).
* Why: splitting dims helps each head represent complementary subspaces; concat+proj recombines signals.
* Shapes: $X:(B,L,d)$ → $Q_i,K_i,V_i:(B,L,d_h)$ for head i, outputs $(B,L,d_h)$ → concat $ (B,L, H d_h)$ → $W^O$ back to $d$.

## KV caching (decoding speed)

* Why: during autoregressive inference, K,V for past tokens remain constant; cache them per layer/head to avoid recomputation.
* Effect: per-step compute reduces to matmul of new Q with cached K (growing length), and new V appended — significant latency reduction.

## Backprop essentials for attention blocks

* Flow: loss → head outputs → A and V → S (via softmax jacobian) → Q,K → linear projections → inputs.
* Key identity to mention: vectorized softmax jacobian used to compute dS from dA efficiently; countless frameworks fuse softmax backward.

## Cross-entropy & softmax (NLL)

* Why: softmax converts logits to probabilities; NLL/CE penalizes low probability on true class, equivalent to MLE for categorical models.
* Must know: gradient identity $dL/dz = p - y$. Log-sum-exp for numerical stability.

## Optimizers (evolution & intuition)

* SGD: baseline, good generalization with schedule. Momentum: accumulates velocity to reduce oscillation.
* AdaGrad: per-coordinate adaptivity but accumulates forever (too aggressive decay). RMSProp: EMA of squared grads (forgetting).
* Adam: m (EMA grads) + v (EMA squared grads) + bias correction → widely used. AdamW: decouple weight-decay from adaptive update → better regularization.
* Interview tip: say “For transformers, AdamW + warmup + decay is standard; for CV, long SGD + momentum + decay often generalizes best.”

## Mixture of Experts (MoE)

* What: replace dense FFN with many experts; gating routes tokens to top-k experts; only selected experts compute → extreme parameter scaling.
* Tradeoffs: huge capacity vs routing complexity (load balancing, latency, cross-device all-to-all comms). Engineering wins matter (DeepSpeed MoE).

## ViT vs Conv-stem & local→global

* ViT patches treat image as sequence — global attention can lack local inductive bias. Conv-stem supplies locality & translation equivariance → often better on small data.
* Hierarchical encoding (local blocks → pooled summaries → global attention) is practical for long / structured inputs.

## Implicit bias / NTK / spectral bias (brief)

* Gradient descent + architecture + initialization bias learning toward low-complexity (low norm / low frequency) solutions.
* NTK regime: very wide nets behave like kernel regression; spectral bias: low-frequency functions learned earlier. PAC-Bayes and MDL connect compressibility to generalization.

## Bayesian perspective (MAP vs Bayes)

* MAP = MLE + prior (regularizer). Full Bayes marginalizes over posterior for predictive uncertainty but is expensive; approximations: VI, SGLD, Laplace, deep ensembles. Use ensembles/SGLD when uncertainty matters.

---

# 3) Common follow-ups & concise answers (quick fire)

**Q: Why mask in decoder self-attn?**
A: To prevent access to future tokens — preserves autoregressivity so the model cannot cheat. Masking sets logits to $-\infty$ before softmax.

**Q: Why softmax over keys (dim=1)?**
A: Each query distributes attention mass across keys; softmax rows correspond to one query’s distribution over keys.

**Q: Why scale by $\sqrt{d_k}$?**
A: Dot products grow with dimension; scaling keeps the variance of scores stable so softmax gradients aren't vanishing/saturated.

**Q: Multi-head vs single big head?**
A: Multiple smaller heads let the model attend in several representational subspaces concurrently—practical empirical benefit and disentangling.

**Q: How does KV caching affect memory?**
A: Cache stores K,V for every past token and every layer/head — memory grows with sequence length. Tradeoff: memory vs latency.

**Q: Adam vs SGD — which generalizes better?**
A: No universal rule. Historically, SGD+momentum with proper schedule often generalizes best for vision; AdamW + warmup is standard & effective for transformers/NLP.

**Q: How does attention help gradients?**
A: Direct path from decoder positions to encoder states avoids long chains of derivatives through recurrent steps; gradients flow to relevant source tokens directly.

**Q: How does teacher forcing affect training?**
A: Training uses ground-truth previous tokens (teacher forcing), enabling parallel training. This causes exposure bias (distribution mismatch) at inference where generated tokens are model outputs.

**Q: Why do we use label smoothing?**
A: Prevents overconfidence, improves calibration and generalization by making the target distribution softer.

**Q: What’s the simplest explanation of MoE?**
A: Many specialists (experts); a small router picks which specialists to run per token → enormous capacity cheaply, but requires routing/load balancing engineering.

---

# 4) Ten 90–120s spoken scripts (rehearse these)

Below are compact scripts you can practice aloud. Each is ~90–120s when spoken slowly, and ends with one short “why it matters” sentence for interviews.

### Script A — Attention (Q/K/V) (90–120s)

“Attention is a mechanism to let one vector query a set of memory vectors and read a weighted sum. We form a query vector $Q$ (what I’m looking for) and compare it to many key vectors $K$ (what each memory slot offers) using dot-product similarity. We scale those dots by $\sqrt{d_k}$ and run a softmax across keys — that gives a probability distribution over memory slots for that query. Finally we do a weighted sum of the corresponding values $V$ to get the readout. In Transformers, Q/K/V are linear projections of the same input sequence. Intuitively, attention lets each position dynamically focus on any other position, removing the single-vector bottleneck of older seq2seq models, and giving a direct gradient path from decoder to encoder states. This is why attention is crucial for long-range dependencies and explainable alignment.”

### Script B — Scaled dot-product formula + backprop sketch (90–120s)

“Scaled dot-product attention computes $S=QK^\top/\sqrt{d}$, $A=\text{softmax}(S)$, $Y=A V$. Backprop flows backwards: from loss into $Y$, we compute gradients w.r.t. $V$ as $A^\top dY$, gradients w.r.t. $A$ as $dY V^\top$; to get gradients into $Q,K$ we pass through the softmax jacobian to get $dS$, and then $dQ=dS\,K/\sqrt{d}$, $dK=(dS)^\top Q/\sqrt{d}$. Practically frameworks fuse softmax+backward for numerical stability. This chain shows attention is differentiable end-to-end and that both keys and values get learned signals."

### Script C — Multi-Head Self-Attention (90–120s)

“Multi-head attention runs several attention computations in parallel. Each head projects input to a smaller subspace $d_h$, computes attention independently, and outputs a head vector. We concatenate heads and linearly project back to model dimension. This gives multiple representational ‘views’ — one head might learn syntax relations, another long-range coreference. The concat+projection mixes these features. Practically splitting helps expressivity while keeping per-head computation small; empirically it improves performance and stability.”

### Script D — Seq2Seq with attention (90–120s)

“In seq2seq for translation, an encoder reads source tokens and produces hidden vectors $h_i$. The decoder generates target tokens autoregressively. At each decode step the decoder computes alignment scores $e_{t,i}=f(s_{t-1},h_i)$, normalizes them to $\alpha_{t,i}$ via softmax, and computes context $c_t=\sum \alpha_{t,i} h_i$. The decoder uses $c_t$ plus previous token to produce next token. Attention removes the compression bottleneck and allows the decoder to focus on relevant source tokens dynamically, increasing translation quality. During training we use teacher forcing; during inference we sample or beam-search over outputs.”

### Script E — KV caching & inference (90–120s)

“During autoregressive inference, each new token requires attention against all previous tokens; recomputing keys and values for past tokens is wasteful. KV caching stores per-layer per-head K and V for past positions so at step t the model only projects the new token’s Q and performs $Q_t\cdot K_{1..t}^\top$ with cached K. This reduces redundant compute and latency for long generations. Memory grows with sequence length, so production systems balance memory and speed; caching is essential for real-time generation services.”

### Script F — Cross-Entropy & softmax (NLL) (90–120s)

“Softmax turns logits into a probability vector by exponentiation and normalization: $p_i=\exp(z_i)/\sum_j\exp(z_j)$. Cross-entropy (or negative log-likelihood) measures how well the predicted distribution matches the true one; for one-hot target t, loss = $-\log p_t$. The gradient w.r.t. logits is simple: $p-y$. This identity makes implementations efficient and numerically stable when combined with the log-sum-exp trick. Cross-entropy equals maximizing likelihood for categorical outputs, so it’s the principled objective for classification and language modeling.”

### Script G — Adam → AdamW (90–120s)

“Optimizers evolved because vanilla SGD struggles with noisy gradients and ill-conditioned loss surfaces. Momentum smooths and accelerates directions; AdaGrad introduced per-coordinate adaptive rates but decays too aggressively. RMSProp fixed that by using EMA of squared grads. Adam unified momentum and RMS into first and second moments with bias correction; it converges fast with little tuning. AdamW decouples weight decay from adaptive scaling, giving true parameter shrinkage consistent with L2 regularization and improved generalization—this is the default for modern transformer training.”

### Script H — Mixture-of-Experts (MoE) outline (90–120s)

“MoE replaces dense feedforward layers with a bank of specialized experts. A small gating network routes each token to the top-k experts; only those experts compute, so each token uses only a small part of total parameters. This gives huge model capacity at modest per-token FLOPs. Challenges: routing fairness (prevent hotspotting), communication (selected experts live on different devices → all-to-all exchanges), and serving complexity. Engineering frameworks like DeepSpeed implement optimized all-to-all kernels and balancing losses to make MoE practical.”

### Script I — ViT vs Conv-stem & hierarchical encoding (90–120s)

“Vision Transformers split images into patches and treat them as sequence tokens — attention mixes them globally. For small datasets this lacks locality bias, so a conv-stem (few conv layers) before patching provides local features and translation equivariance. Hierarchical encoding (local windows → pooled summaries → global attention) reduces compute and preserves multiscale structure. In short: use conv-stem or hierarchical ViT when data is small or locality matters; pure ViT shines with large datasets and pretrained models.”

### Script J — Implicit bias / NTK / spectral bias (90–120s)

“Modern SGD training doesn’t pick any solution uniformly — it has implicit biases. In some regimes (very wide networks), training behaves like kernel regression under the Neural Tangent Kernel (NTK). Across many models, gradient descent learns low-frequency (smooth) functions earlier — called spectral bias — which explains why networks generalize to simple patterns before memorizing noise. These ideas link to MDL/Kolmogorov simplicity: networks prefer simple programs/solutions unless the data forces complexity. For interviews: phrase it as ‘training dynamics + architecture induce a simplicity bias that often leads to good generalization’.”

---

