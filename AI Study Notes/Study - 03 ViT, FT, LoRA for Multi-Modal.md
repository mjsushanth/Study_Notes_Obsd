

---
1. First, get a **compact but deep transformer/ViT = geometry/function space** recap — so all the LoRA talk has a clear base.
    
2. Then, a **math-heavy LoRA explanation** that unifies:
    - “LoRA as ΔW = A B”
    - “LoRA modifies Q/K/V/Proj”
    - “LoRA bends geometry / manifolds / attention patterns” in one coherent picture.


## 1. Transformers / ViT as Geometry in Embedding Space

- At any point, a transformer layer takes: - a sequence of token vectors, shape: X ∈ R^{L × d} (L tokens, each d-dim).
- For ViT:  `tokens = CLS + 196` patch tokens, X is `[197, 768]` after patch embedding + pos embedding.
- Any projection we see (e.g. `W_q`, `W_k`, MLP layers) is just - Operation: `Y = X W` where rows of X are tokens, each multiplied by W.

> So “transformer as geometry” = **apply many linear maps and mixing operators, layer by layer**. Each layer rotates, stretches, and repositions the sequence of vectors in R^d.


### Self-attention as “learned kernel” over tokens

For a single head:
- `Q = X W_q` (queries)
- `K = X W_k` (keys)
- `V = X W_v` (values)
Shapes:
- `Q, K, V ∈ R^{L × d_h}` (d_h = head dim)
Scores:
- `S = Q K^T / sqrt(d_h)` → `R^{L × L}` (token-to-token scores)

Attention weights:
- `A = softmax(S, dim=-1)` → each row of A is a probability distribution over which tokens to look at.
Context:
- `O = A V` → `R^{L × d_h}`

Then heads are concatenated, projected by `W_o`, etc.

The critical point: **all nonlinear structure in attention is driven by linear projections** (Q/K/V matrices) plus softmax. When we change `W_q` / `W_k` / `W_v`, we change:

- which directions in the embedding space are considered “queries/keys/values”
- how tokens see each other
- which patterns become salient

This is the exact place where LoRA intervenes.

---
## LoRA: The Mathematical Core


**Zoom** in on **one** matrix first, then generalize.

- Single layer baseline: Pick any learned linear map, say `W_q ∈ R^{d × d}`.
- Input: token vector `x ∈ R^d` , Output: `q = x W_q` (row-vector × matrix convention)
- Function from x to q is: `f(x) = x W_q`
- This is a point in **function space**: all possible linear maps R^d → R^d form a space of dimension d^2 (each weight is a coordinate). Training normally moves W_q in that d^2-dimensional parameter space.

**LoRA** **decomposition**

LoRA says: instead of freely modifying W_q, we **freeze W_q** and add a **low-rank delta**:
- `W_q' = W_q + ΔW_q`
and define:
- `ΔW_q = A_q B_q`
where:
- `A_q ∈ R^{d × r}`
- `B_q ∈ R^{r × d}`
- r << d (e.g. r = 4, 8, 16)

So the new function is:
- `f'(x) = x W_q' = x (W_q + A_q B_q) = x W_q + x A_q B_q`

We train **only** A_q and B_q.
- `rank(ΔW_q) ≤ r` — we force the update to lie in a **low-rank subspace** of all possible updates.

> **ΔW = A B view**.

---

### How it modifies geometry

Look at the output:
- baseline: `q = x W_q`
- with LoRA:  
    `q' = x W_q + x A_q B_q`
Define:
- `z = x A_q` → `R^r`
- `q_delta = z B_q` → `R^d`

So LoRA does:

> 1. Project x down to a low-rank “adapter space” R^r via A_q.
> 2. Re-expand that low-dimensional feature back into R^d via B_q.
> 3. Add this tweak onto the old q.

a **bottleneck adapter** in linear form.

----

## LoRA on Q/K/V and Attention

Now, connect “ΔW = A B” to **attention mechanics**.
We apply LoRA not just to W_q, but typically to:

- `W_q` (query projection)
- `W_k` (key projection)
- `W_v` (value projection)
- sometimes `W_o` (output projection) or MLP projections

So for Q/K/V:
`W_q' = W_q + A_q B_q W_k' = W_k + A_k B_k W_v' = W_v + A_v B_v`

Then:
- `Q' = X W_q' = X W_q + X A_q B_q`
- `K' = X W_k' = X W_k + X A_k B_k`
- `V' = X W_v' = X W_v + X A_v B_v`

Original scores:

- `S = (X W_q)(X W_k)^T / sqrt(d_h)`  
    where `S[i,j] = <q_i, k_j> / sqrt(d_h)`

With LoRA:
- `Q' = Q + X A_q B_q`
- `K' = K + X A_k B_k`

So:
`S' = Q' K'^T / sqrt(d_h)`  
`= (Q + X A_q B_q) (K + X A_k B_k)^T / sqrt(d_h)`

> - Each attention score `S'[i,j]` is influenced by low-rank corrections that define **new directions** in which queries and keys can move. !!

> - q_i and k_j are **repositioned** in the embedding space
> - inner products `<q_i, k_j>` change
> - attention weights A'[i,j] = softmax(S')[i,j] re-focus on different tokens/patches.


### How context vectors change:
Context:
- baseline: `O = A V`
- LoRA: `V' = V + X A_v B_v`, `A'` from Q'/K' as above.

So O' ≈ `A' V'` incorporates:
- **where** we attend (due to Q'/K')    
- **what** we aggregate from those attended patches (due to V')

---

Vanilla ViT encoder **does not know anything about text**.
- It is trained only for image classification.
- Its patch embeddings encode visual semantics _only within the image world_.
- There is no concept of “dog” ↔ “this patch,” “child” ↔ “these regions.” 
### Exact build up into Cross-Modality

- text tokens → semantic “query vectors”
- image patches → semantic “key/value vectors”
- MaxSim(text_token, patch) to reflect **true cross-modal alignment**

Full fine-tuning is not feasible. total: ~86M+. memory expensive.

### Core Idea of LoRA, in Intuition-First Terms

- **Tiny, low-rank “delta” that gently nudges a frozen model into a new behavior.**
- It does this by **injecting tiny trainable matrices** into specific linear layers while keeping the original massive layers **frozen**.
- You only optimize _hundreds of thousands_, not _tens of millions_ of parameters.

In a transformer, attention layers contain matrices like:

```
	W_q  (query projection) 
	W_k  (key projection) 
	W_v  (value projection) 
	W_o  (output projection)
	
	Each is a `D × D` linear projection.
```

During inference/training, the model does:
	`h_out = h_in @ W_q`

LoRA modifies it into:
	`h_out = h_in @ (W_q + ΔW_q)`

but crucially, **ΔW_q is low-rank**: `ΔW_q = A @ B`
LoRA adds an adaptation matrix with **rank r** instead of rank D.

```
With r = 16:

W_q:   768 × 768  → ~590k params
ΔW_q:  768×16 + 16×768 → ~24k params
```

Neural layers usually have **massive unused capacity**.  LoRA exploits this by modifying the _directionality_ of transformations without needing to rewrite the entire weight matrix.

---

### Why LoRA Is Perfect for ViT → Text Alignment

In late interaction, you care about:
- **patch embeddings** (before pooling)
- **attention patterns inside the transformer blocks**

Goal is not classification accuracy.  It is **alignment of image patches to text tokens**.

1. **Attention heads to shift how they combine patches**
2. **Patch embeddings to move toward the text embedding space**

> What we need to recognize here is, that LoRA modifies exactly the right spots. 

That is, 

> LoRA to:
> 	- `qkv` linear layers in attention    
> 	- `proj` layers in attention output
> 	- sometimes MLP layers

We are specifically influencing **how patches attend to each other, which patches become semantically salient, how global meaning is distributed across the patch grid, how patch embeddings orient relative to text embeddings**. 

**LoRA shapes the geometry** of patch-token embeddings. It “bends” the ViT's learned manifold so that:
- text directions become meaningful for image patches
- semantic categories align across modalities
- discriminative patches pop out
- background patches recede

---

---

## **What LoRA Is Actually Doing**

Think of ViT : linear/attention transformations with millions of degrees of freedom frozen.
LoRA inserts tiny **steering wheels**.

By training only these steering wheels, the huge frozen model can:
- reshape patch geometry
- create semantically meaningful attention patterns
- align vision patches with text tokens
- maintain generalization and stability
- avoid catastrophic forgetting

LoRA is not doing “small training.”  

> It is doing **precise geometric structuring, manipulation, surgery** on the model’s latent space with minimal parameters.

## Why This Is Enough to Align ViT Patches with Text

**We have:**
- Image patches: `patches = ViT_patches(image)` → `[P, D]`
- Text tokens: `tokens = CLIP_text(caption)` → `[T, D_text]` → projected to `[T, D]`
- Late interaction: `MaxSim(tokens, patches)`
**And:**
- ViT patches live in a **pure vision semantic space**.
- CLIP text tokens (after projection) are forced into that space, but the space was never trained to align with them.
**We want:**
- For each token “dog”, “snow”, “baby”, etc., some subset of patches to share a strong direction.
**LoRA is attached to the ViT:**
- In each block, on Q/K/V (and possibly MLP).

During training, we minimize contrastive loss:
- `Loss = contrastive(MaxSim(text_i, image_i), MaxSim(text_i, image_j!=i))`
Gradient flows:
1. Through *MaxSim → patch embeddings*
2. Through *patch embeddings → attention layers* and MLP layers
3. Through those *layers → A_q, B_q, A_k, B_k, A_v, B_v*

> The only way the model can satisfy the loss is by adjusting **the low-rank “bending” matrices** A,B so that patch directions align with text directions.

This is the **geometric interpretation**:
- The base ViT defines a manifold M in R^D where patches lie.
- LoRA adds low-rank ΔW layers that **warp the manifold** slightly so that semantic axes line up with text axes.

> We dont really write “align manifolds” in equations, but it’s exactly what’s happening: the low-rank updates constantly nudge the patch embeddings so that inner products with text embeddings reflect semantic similarity.

This is the **function-space view**:
- Instead of searching over all possible functions f_θ, we search over a **low-dimensional slice** of that huge function family.
- This slice is constructed so that it can still significantly rotate and reshape embeddings in useful directions.

Read later:
https://www.linkedin.com/posts/aayush-sugandh-785181190_i-worked-out-the-math-behind-dyloradynamic-activity-7395793211530616832-_IX6/

---

## **Summary Mental Model**

- **Transformers = geometry machines.**
- **ViT patch embeddings = spatial geometric tokens.**
- **Late interaction = geometric matching (token → patch).**
- **LoRA = low-rank geometric correction tools.**
- **Cross-modal alignment = reshaping embedding geometry so text directions correspond to patch directions.**