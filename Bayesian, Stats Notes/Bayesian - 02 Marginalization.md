
![[Pasted image 20251206101910.png]]
![[Pasted image 20251206101930.png]]


## Building General Intelligence: A Probabilistic Perspective

- **Talk Structure:**
    - Part 1: Learning and generalization from the perspective of Probability Theory, Bayesian Inference, and Compression.
    - Part 2: Formalizing compression using Kolmogorov complexity, PAC-Bayes bounds, and finite hypothesis bounds to argue for the possibility of general intelligence.
    - Part 3: Discussion on how to build such systems.
- **Core Thesis:**
    - It is possible to build broadly, generally intelligent systems, in contrast to what might be suggested by the No-Free-Lunch theorems.
    - However, we are far from achieving it; estimates 50-100 years for AI capable of proposing theories like general relativity or quantum mechanics.
    - Emphasizes the importance of AI safety and alignment.

---


### **Spectral bias** story: not “GD is secretly Newton” 

but “GD on a quadratic decouples into independent exponential decays along eigen-directions; high-eigenvalue directions (smooth modes) dominate early.”

if you look at what eigen-directions correspond to in function space (e.g., for kernels / NTK):
- Large λ modes are typically **smooth / low-frequency** components of the target function.
- Small λ modes correspond to **high-frequency, wiggly** components.****

So gradient descent naturally:
> fits low-frequency (simple) structure first, and only later starts using high-frequency (complex) modes.


This already gives you simplicity bias _without_ any second-order algorithm. You’re using only the gradient; the Hessian shows up in the _dynamics_, not because you invert it, but because it’s the curvature controlling speeds of different modes.

----

### Implicit bias of gradient descent: minimal-norm **solutions**


- **Linear regression with squared loss** and many global minimizers:  
    Gradient descent from small initialization converges to the **minimum ℓ₂-norm** solution (the pseudoinverse solution).

- **Logistic regression on linearly separable data**:  
    Gradient descent doesn’t stop at any finite norm; the weights diverge, but the direction converges to the **max-margin separator**.

Neither of these use the Hessian explicitly. The solution you land on is determined by:
- Parameterization, Initialization, The particular flow of gradient descent.