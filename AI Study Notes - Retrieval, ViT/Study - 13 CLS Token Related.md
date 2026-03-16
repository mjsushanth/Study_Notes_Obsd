
**Takeaway:** `CLS` is not a “position encoding token.” It’s a **learnable aggregation token**: you prepend a trainable vector to the token sequence, and you _force downstream objectives to read from it_, so training pressure makes it become a useful “summary.” When that downstream pressure is weak/misaligned (common in self-supervised vision), `CLS` can look noisy or “non-semantic,” which is exactly the critique you’ve seen.

## 1) The origin story: why `CLS` exists at all

### In NLP (BERT era): “I need one vector for whole-sequence tasks”

Transformers natively output **one vector per token**. But tasks like sentiment, topic, entailment, “is sentence A related to sentence B,” need **one vector for the entire sequence**.

BERT’s move was: prepend a special token `[CLS]` and define the classifier as `head(h_cls)`. Then the model can learn to use attention layers to route global information into that one slot.

Mechanically, it’s just:

- tokens: `[CLS], w1, w2, ..., wT`
    
- encoder outputs: `h_cls, h1, ..., hT`
    
- for classification: `logits = W * h_cls + b`
    

So: **`CLS` is a learnable “sink” for global info because the loss reads from it.**

### In ViT (Vision Transformer): “patches are tokens; I still want one vector”

ViT treats each image patch like a token. Same issue: classification needs one vector. So it copies the BERT trick: prepend a trainable `cls_token`, add position embeddings, run the transformer, then read out token 0 for classification.

Your own ViT code spells this out very explicitly: `use_cls_token`, create `self.cls_token = nn.Parameter(...)`, prepend it, and later “extract CLS token or mean pool.”

## 2) What `CLS` is doing inside attention (the real mental model)

Let the sequence be length `N = 1 + num_patches` (or `1 + num_words`). In each transformer layer, every token updates by attending to all tokens:

`x_i <- x_i + sum_j softmax(q_i·k_j / sqrt(d)) * v_j`

Now focus on token `i = CLS`. That token has its own query `q_cls`, so it chooses **what** to aggregate from the rest. Over layers, it becomes a **learned pooling operator**:

- early layers: collects low-level cues
    
- mid layers: mixes parts / relations
    
- late layers: becomes whatever the loss needs (class-discriminative, instance-discriminative, etc.)
    

Key point: there is _no guarantee_ that `CLS` will be “similar” to patch tokens in cosine space. It’s not a literal average. It’s a **content-addressed, task-shaped pooling**.

## 3) Quick code templates (NLP + vision)

### A) BERT-style `CLS` usage (HuggingFace)

```python
# pip install transformers torch
import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "a dog is running in the snow"
batch = tok(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    out = model(**batch)  # out.last_hidden_state: [B, L, D]

h_cls = out.last_hidden_state[:, 0, :]      # [B, D]  (token 0 is [CLS])
h_mean = out.last_hidden_state.mean(dim=1)  # [B, D]  (simple baseline pooling)

print(h_cls.shape, h_mean.shape)
```

### B) ViT-style `CLS` usage (PyTorch / timm)

```python
# pip install timm torch
import torch
import timm

m = timm.create_model("vit_base_patch16_224", pretrained=True)
m.eval()

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    # Many timm ViTs expose forward_features
    feats = m.forward_features(x)  # often returns [B, D] already (CLS-pooled)

print(feats.shape)
```

### C) “Similarity between CLS and patches” (what you were reading about)

If you have per-token outputs `H: [B, N, D]` where token 0 is CLS and the rest are patches:

```python
import torch
import torch.nn.functional as F

# H: [B, N, D]
cls = F.normalize(H[:, 0, :], dim=-1)        # [B, D]
patch = F.normalize(H[:, 1:, :], dim=-1)     # [B, N-1, D]

# cosine similarities between CLS and every patch
sim = torch.einsum("bd,bnd->bn", cls, patch)  # [B, N-1]

# you can reduce however you want
sim_mean = sim.mean(dim=1)   # [B]
sim_max  = sim.max(dim=1).values
```

That screenshot question you shared is basically asking whether someone did “CLS vs all patches then average.” The important thing: **there isn’t one canonical reduction**; different papers pick mean/max/top-k/attention-weighted, depending on what they’re measuring.

## 4) Why DINO `CLS` often doesn’t cluster like CLIP (and why you shouldn’t expect it)

Your intuition was: “If CLS represents the whole image, same-class images should cluster; and CLS should be similar to its own patches.” That expectation is reasonable for **supervised classification** or **text-supervised alignment** (CLIP). It is _not guaranteed_ for DINO-style SSL.

Here’s the core reason:

**CLIP** forces global semantics because the objective aligns image embeddings to **language** (which carries category-ish abstraction). So the global embedding is pressured to be semantically organized.

**DINO** (self-distillation) pressures the representation to be **instance-discriminative / view-invariant** under augmentations, not necessarily “class-separable.” It can encode fine-grained cues, texture/style, or augmentation-invariant shortcuts that do not map cleanly to your label taxonomy. So `CLS` can be excellent for nearest-neighbor retrieval under certain invariances yet not form crisp class clusters in a naive 2D projection.

Also: many DINO/DINOv2 use cases rely heavily on **patch tokens for dense correspondences**; the global token is useful, but the _most semantically stable_ signal for many downstream tasks may come from pooling patch tokens (mean/attention pooling) rather than trusting `CLS` as “the” representation.

## 5) The critique of `CLS` (when it fails, and what people do instead)

The criticisms are basically “`CLS` is a single bottleneck with no physical meaning unless the loss forces it.”

Common failure/fragility modes:

1. **Bottleneck + under-trained aggregator**: if your objective doesn’t read from `CLS` strongly, the model has little incentive to make it a good global summary.
    
2. **Optimization quirks**: because `CLS` is one vector that must capture everything, it can become sensitive to augmentation policy, training recipe, and normalization.
    
3. **Mismatch to dense information**: for vision tasks requiring spatial detail, `CLS` can discard structure; patch pooling can retain more.
    

Hence the very standard alternative you already have in your codebase: _either_ extract `CLS` _or_ do mean pooling over tokens (“extract CLS token or mean pool”).

Practical replacements people use:

- **Mean pooling of patch tokens** (often surprisingly strong).
    
- **Attention pooling** (learn a pooling query, sometimes multiple queries).
    
- **Concatenate / pool multiple layers** (late-layer CLS can be too task-shaped; mid layers can be nicer).
    
- **Use patch tokens directly** for similarity/correspondence (especially for DINO-style models).
    

## 6) A “picture template” of what’s happening

Think of the token sequence as a set of slots:

`[ CLS | p1 | p2 | ... | pK ]`

Each layer is “everyone reads everyone.” CLS is special only because:

- it has a special initial embedding
    
- it’s placed in a consistent index (0)
    
- your downstream head chooses to read it (and thus backprop shapes it)
    

So `CLS` is best seen as a **learned query for pooling**, not as a magical “whole image vector” that must resemble the parts.

## 7) A tight experiment you can run to settle this in your head

Use any ViT/DINO model that returns token embeddings and compare 4 embeddings per image:

- `CLS_last`
    
- `mean_patch_last`
    
- `CLS_mid` (say layer 6)
    
- `mean_patch_mid`
    

Then evaluate: “do same-class images cluster” under each embedding? You’ll usually find:

- CLIP: `CLS_last` clusters decently.
    
- DINO: `mean_patch` often behaves more smoothly than `CLS_last` for coarse semantics.
    

If you want, paste the exact model you’re using (e.g., `dinov2_vitb14`, `dino_vitb16`, etc.) and whether you’re reading _pre-norm_ or _post-norm_ tokens; I’ll give you a minimal, correct extraction snippet for that specific architecture.