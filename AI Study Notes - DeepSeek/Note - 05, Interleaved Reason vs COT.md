
![[image-18.png]]


![[image-19.png]]



- What's truly amazing is that it was trained in 4 bits (INT4) using a technique known as QAT (Qantization Aware Training). This technique simulates the weights that can only be loaded on enterprise-level GPUs like the H100... with Int4 you get THE SAME ACCURACY as the giant models without requiring enterprise GPUs. Even Turing architecture GPUs (RTX 2060-RTX 2080 Ti) have Tensor Cores to perform inference in Int4, the problem with the consumer GPUs is they don't have enough VRAM to load a 600GB model. If the RTX 2080 Ti had 600GB of VRAM, you could load Kimi K2 and solve 42% of all text-problems in Humanity's Last Exam.

### Kimi K2 Thinking -- MOE + QAT 

![[image-20.png]]

