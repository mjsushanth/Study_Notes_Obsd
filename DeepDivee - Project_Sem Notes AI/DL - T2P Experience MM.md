
```
1. Mental Model: What is Text-to-Pose Generation?
   └─ Analogy, intuition, problem framing

2. Phase 1: Baseline Diffusion (The Catastrophic Failure)
   ├─ What went wrong (10^15 loss)
   ├─ Why normalization space matters
   └─ Lessons learned

3. Phase 2: Anatomical Awareness (The Breakthrough)
   ├─ Forward kinematics intuition
   ├─ Denormalization discovery
   └─ Why constraints improve generalization

4. Phase 3: Text Conditioning (The Bridge)
   ├─ Cross-attention mental model
   ├─ LayerNorm vs GroupNorm decision
   ├─ Classifier-free guidance geometry
   └─ Progressive guidance scaling

5. Architecture Deep Dive
   ├─ UNet-Transformer bridge (the tension)
   ├─ Residual block orchestration
   ├─ Time embedding injection
   └─ CLIP projection layer (semantic space bridging)

6. Training Dynamics
   ├─ SGLD intuition
   ├─ Noise calibration
   ├─ Gradient clipping rationale
   └─ Loss component analysis

7. Dataset Processing
   ├─ First-action segmentation (POS-tag algorithm)
   ├─ Cluster-aware sampling (why it matters)
   ├─ 22-joint vs 66-dim explanation
   └─ Coordinate system handling

8. Key Insights & Patterns
   ├─ "The Big Insights"
   ├─ What actually matters in practice
   ├─ Common antipatterns
   └─ Troubleshooting guide

9. Interview Cheat Sheet
   ├─ Fast recap
   ├─ Common follow-ups
   └─ Key metrics to remember

10. Comparison Tables
    ├─ Phase 1 vs 2 vs 3 results
    ├─ Architecture alternatives
    └─ When to use what
```

---
