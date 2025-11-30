
---

## Summary Mental Model

**Pooling = global, homogenous, classification-focused representation**  
**Patch tokens = local, heterogeneous, retrieval-focused representation**  
**Late interaction = token–patch geometric matching**  
**CLS token = semantic sink that suppresses patch autonomy**  
**Dropping CLS + removing pooling = unlocking fine-grained alignment**

---

## 1. Patchification: the “tokenization” of vision

A 224×224 image is converted into:
- 14×14 = **196 patches**
- Each 16×16 patch → flatten → linear projection → **768-d vector**
```
Tokens: [CLS] + [patch_1, patch_2, ..., patch_196]
Shape: [B, 197, 768]
```

## 2. CLS token: the “summary token”

CLS is:
- A _learnable vector_
- Inserted at position 0
- Intended to aggregate information from all patches
- Used as the **global image representation** after attention layers

Why it works:  
- Self-attention lets CLS attend to all patches, and the patches attend back. CLS becomes a “collector.”
- In late interaction, we **ignore CLS** because we don’t want global pooling.

## 3. Positional embeddings: recovering spatial identity

Without position information, the transformer would treat patch order as irrelevant.

ViT adds a learnable matrix:
`PosEmb: [197, 768]`

Added element-wise to the tokens:
`tokens = tokens + PosEmb`

This gives each patch a sense of “where it is in the image.


## 4. Multi-Head Self-Attention

```
Q = tokens * W_Q   # [B, 197, D]
K = tokens * W_K   # [B, 197, D]
V = tokens * W_V   # [B, 197, D]

Attn = softmax(Q K^T / sqrt(d_k))          # [B, 197, 197]
out  = Attn V                               # [B, 197, D]
```

Key intuition:
- Each token looks at (attends to) all other tokens.
- CLS learns to collect global info.
- Patch tokens learn contextual understanding (shadows, edges, shapes, etc).

## 5. “Collapse”: pooling vs sequence output

(A) With pooling → single vector `[B, 768]`
(B) Without pooling → full sequence `[B, 197, 768]`

- When we apply **LoRA**, we will **inject low-rank updates into W_Q and W_V**.
- LoRA modifies **how tokens talk to each other** — which can drastically change how specific patches align with text tokens.

## 6. Some CLS - pooling confusion:

**CLS is not the same thing as pooling**, and indeed pooling is the real “killer.”  
But **CLS _still does_ interfere with late interaction**, not because it removes shape, but because it warps the similarity geometry.

> “Pooling is the real thing we must disable. CLS doesn’t remove patch vectors.”

If pooling is applied → you lose all patch vectors → late interaction dies.

**If pooling is disabled (so patch vectors survive), why _still_ remove CLS?  Isn’t CLS just another token with dimension 768?**

> CLS is _trained_ to be the **single-vector representation of the whole image**. Its purpose is _explicitly_ to **absorb**, **summarize**, and **pull** information from patches.

- Late interaction: `score = Σ_i max_j dot(text_token_i, patch_vector_j)`
- This mechanism assumes that **every vector j corresponds to a _local region_ of the image**.

So when text token = “dog” → the model finds the patch representing dog.  
When text token = “grass” → another patch.  
When text token = “sky” → another patch.
This gives **token–region grounding**.


- CLS represents the **entire image**, because during training: - Gradients propagate through CLS to learn:  **“summarize the whole visual content into this vector.”**

- So if CLS is included in the token set for late interaction: Every text token tends to max-align with CLS — not patches. Because CLS has absorbed all features. 

Late interaction should becomes a cross-token, cross-patch alignment and should reflects real spatial relationships. Removing CLS restores spatial semantics.

- Pooling destroys all patch tokens → breaks late interaction completely.
- CLS **overshadows** them in similarity space ` max_j dot(q_i, token_j) → often chooses CLS`


To understand deeper do these lol:

- Visualizations of dot-product distributions
- Small simulation where CLS massively dominates patch similarity