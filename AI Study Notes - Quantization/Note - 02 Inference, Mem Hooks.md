
| Concept      | Memory hook                                                        |
| ------------ | ------------------------------------------------------------------ |
| **Weights**  | Static learned numbers. Quantization mostly shrinks these.         |
| **KV cache** | Dynamic memory of previous tokens. Long context makes this grow.   |
| **Prefill**  | Processes the prompt. Controls time-to-first-token.                |
| **Decode**   | Generates one token at a time. Controls tokens/sec.                |
| **VRAM**     | Best place for weights/cache because it is closest to GPU compute. |

---

|#|Question|Strong answer|
|--:|---|---|
|1|**A 13B INT4 model fits in 12 GB VRAM, but slows/crashes at 32k context. Why?**|The **weights fit**, but the **KV cache grows with context length**. At 32k tokens, every layer stores many more K/V vectors, so VRAM gets consumed by dynamic runtime memory, not just static model weights. Also, during decode, each new token attends over a much longer cached history, so per-token work increases.|
|2|**Why can FP16 be faster than INT4 on some GPUs?**|INT4 is smaller, but it is not automatically faster. FP16/BF16 may have extremely optimized tensor-core kernels. INT4 weights often need unpacking, scaling, and specialized fused kernels. If the runtime/hardware path is weak, dequantization overhead can erase the memory-bandwidth savings.|
|3|**What is the difference between “model loaded” and “model runs well”?**|“Loaded” only means the runtime allocated enough memory to place the model and start execution. “Runs well” means there is enough extra memory for KV cache, buffers, CUDA workspace, long context, and stable device execution without paging, CPU/GPU transfer bottlenecks, or thermal throttling.|
|4|**Why is long context expensive twice?**|First, **prefill** must process more prompt tokens, so time-to-first-token increases. Second, the **KV cache is larger**, so decode has more stored keys/values to read during attention. Long context therefore increases both initial latency and ongoing per-token cost.|
|5|**What is prefill vs decode?**|**Prefill** processes the entire input prompt and builds the initial KV cache. It is more parallel because many prompt tokens can be processed together. **Decode** is the autoregressive loop: generate one token, append its K/V to cache, generate the next token. Decode is sequential because each token depends on the previous one.|
|6|**Why does GPU offload help, but CPU offload can slow down?**|GPU offload helps because transformer inference is dense tensor algebra, and GPUs have high parallelism plus high-bandwidth VRAM. CPU offload slows when some layers stay on CPU because hidden states must move between CPU RAM and GPU VRAM. The transfer and synchronization cost can dominate.|
|7|**Why is KV cache different from model weights?**|Model weights are **static learned parameters** loaded once. KV cache is **dynamic runtime memory** created from the current prompt and generated tokens. Weights scale with model size; KV cache scales with context length, number of layers, KV heads, precision, and batch/concurrency.|
|8|**Why does quantization reduce memory but not eliminate inference difficulty?**|Quantization shrinks the **weights**, reducing capacity and bandwidth pressure. But inference still needs KV cache, temporary buffers, attention reads, dequantization, kernel execution, sampling, and possible CPU/GPU transfers. Quantization solves the biggest static memory problem, not the entire runtime system problem.|
|9|**Why does batching help servers more than local chat?**|Batching lets servers process many requests together, creating larger matrix operations and better GPU utilization. But local chat is often batch size 1, especially during decode. One user’s token-by-token generation has limited parallelism, so batching benefits are smaller unless multiple sequences are processed together.|
|10|**What is the most complete mental model of LLM inference?**|LLM inference is a repeated pipeline: **tokens enter → embeddings → transformer layers → attention/MLP → KV cache update → logits → sample next token → repeat**. The bottleneck is often not “thinking,” but moving huge weights and KV cache through SSD, RAM, VRAM, GPU caches, and optimized kernels fast enough.|