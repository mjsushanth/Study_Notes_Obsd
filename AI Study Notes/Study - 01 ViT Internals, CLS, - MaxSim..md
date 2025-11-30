
- Earlier- Wrote some practice code to explore PyLate, then Late-Interaction Retrieval and Peft LORA. 
- The research i was attempting with quick mini project codes was: how do we study *vision modality*, _images_, _local patch embeddings_, and _multi-vector retrievals ??
- **ColPali** - biggest inspiration here. Applies late interaction to vision-text retrieval, directly embedding document images without OCR. [ https://huggingface.co/blog/manu/colpali ]
- Built on PaliGemma-3B (Vision-Language Model) + ColBERT's late interaction mechanism
- Multi-modal retrieval is HOT rn. i think ?? :) 

> **After some basic code, we got**:
	`Loaded vision model: vit_base_patch16_224 
	`Vision encoder feature dim: 768 `
	`Input batch shape: torch.Size([2, 3, 224, 224]) `
	`Patch embeddings shape: torch.Size([2, 196, 768])`

### Study the results: Pause.

- shape [2, 768]: we are getting an item of 'pooled embedding per image', not really patch-token grid we actually want for late interaction.
- image_transform turns a PIL image into a normalized tensor: given that (H, W, 3) RGB input, After resize + ToTensor: [3, 224, 224], After batching in the DataLoader: [B, 3, 224, 224].
- check with a simple print, observe, `torch.Size([2, 3, 224, 224]).`

### Vision Transformer Patch Embeddings:

- For ViTs in timm, this model does:
  - Split each image into non-overlapping 16×16 patches.
  - For 224×224, that’s a 14×14 grid → 196 patches.
  - Flatten each patch and multiply by a learned matrix W_patch to get D=768-dim patch tokens: patches: [B, 196, 768].
  - Prepend a learnable CLS token: tokens: [B, 197, 768].
  - Add positional embeddings, then pass through a stack of Transformer encoder blocks.
  - Pool the final tokens to produce a single [B, 768] embedding.
- above, the code just creates the model with num_classes=0 but left the default global_pool behavior, timm gives a single global feature vector per image, shape [B, 768].


### What ViT is doing internally:

Input image 
-> Divide into 14×14 grid of 16×16 patches 
	-> 196 patches -> Flatten each patch (3×16×16 = 768 scalars)
-> multiply by W_patch ∈ R^{(3*16*16)×768}. -> patch token matrix (P)
-> Add CLS token: learnable CLS vector c ∈ R^{768}: tokens = concat([CLS], P) → [B, 197, 768].
-> Add positional embeddings E_pos ∈ R^{197×768}
-> Transformer encoder blocks ( Each layer applies multi-head self-attention and MLP to the token sequence )
	-> Internally, attention splits 768 → H × d_head (e.g., 12 heads × 64 dim)
	-> `Q = tokens * W_Q, K = tokens * W_K, V = tokens * W_V, A = softmax(Q K^T / sqrt(d_head)), output = A V`
-> Pooling -> Single [B, 768] embedding per image.
	-> Take the CLS token only [B, 1, 768] → [B, 768], or mean-pool all tokens.

> Code's Core Modification: **For late interaction, we do not want this final pooling. We want the pre-pooled [B, 197, 768] (and then drop CLS to get [B, 196, 768]).**

  

### Why this matters for late interaction:

- Late interaction works by matching each text token to all image patches and then aggregating:
  - Text side: text_tokens: [B, T, D]
  - Image side: patch_tokens: [B, P, D]
- If you only have one vector per image [B, 768], there is nothing to “late interact” with; it degenerates to a regular cosine similarity between two global vectors.
- Patchwise [B, P, D] is what unlocks:
  - Token–patch alignment heatmaps, Fine-grained grounding (“this token matched that region”), More expressive ranking than global pooled embedding.



