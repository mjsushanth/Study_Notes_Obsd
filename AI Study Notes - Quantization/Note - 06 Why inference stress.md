why inference stresses the machine.

## 1. Inference is data movement, not just computation

- LLM generates text, it does not produce the whole answer in one shot. It generates **one token at a time**. For every new token, the model runs through all transformer layers again: attention, MLP, normalization, output logits, sampling.

- read weights → read/write KV cache → run matrix ops → sample next token → repeat
- Can the machine move the model weights and KV cache fast enough every token?


## 2. The memory hierarchy: SSD, RAM, VRAM, GPU cache

- SSD/disk → OS cache → RAM → VRAM → GPU cache/registers → compute cores

```
model weights + KV cache + runtime buffers fit in VRAM
okay - some layers in VRAM, some in RAM/CPU
bad - RAM is also tight, OS starts paging to SSD
```


## 3. Capacity vs bandwidth: two different problems

```
Can the model fit?
How fast can the model’s bytes be moved during generation?
```

## 4. Prefill vs decode: the two phases of inference

- Prefill is when the model processes your prompt. prompt tokens → transformer → KV cache created.
- Decode is when the model generates the answer one token at a time. generate token 1 → generate token 2 → generate token 3 → ...

```
total latency ≈ prefill time + decode time

total latency ≈ cost(prompt length) + cost(output length)
```


## 5. KV cache

Without KV cache, generation would be extremely wasteful.
With KV cache, the model can reuse previous internal states. But memory grows with:

```
layers * context length * KV heads * head dimension * bytes * batch size
```

This is why a model can run fine at 2k context but struggle at 16k or 32k context.


## 6. CPU/GPU offloading: why partial fit is slower

layers 0–25 on GPU
layers 26–31 on CPU

cleanest - all layers on GPU + KV cache in VRAM
hybrid slow - some layers on GPU + some on CPU + transfers between them

-----------

**`mmap` / memory mapping:** Lets the OS map a model file into virtual memory instead of eagerly copying the entire file. Helpful for loading and CPU inference, but it does not make disk as fast as RAM or VRAM.

**Temporary buffers:** Inference needs more than weights: hidden states, attention buffers, logits, sampling buffers, quantization scales, CUDA workspace, and KV cache. This is why a 10 GB model may need more than 10 GB to run.

**OS paging/swapping:** If RAM is overfilled, the OS moves memory pages to disk. For LLM inference, this can destroy performance because tensor access becomes disk-bound.

**Compute-bound vs memory-bound:** Compute-bound means arithmetic is the bottleneck. Memory-bound means moving data is the bottleneck. Decode is often memory-bound because the model streams huge weights for one token at a time.

**Time to first token:** Mostly affected by prompt length and prefill. Long prompt means slower first visible response.

**Tokens per second:** Mostly affected by decode speed, model size, quantization, KV cache reads, runtime kernels, and hardware bandwidth.

**Batching:** Processing multiple users/sequences together improves GPU utilization. Great for servers, less relevant for single-user local chat.

**Continuous batching:** A server trick where finished requests leave and new ones enter dynamically, keeping GPU utilization high.

**Paged attention:** A memory-management trick for KV cache. It stores cache blocks like pages to reduce fragmentation and support many variable-length requests.

**FlashAttention:** An optimized way to compute attention with less memory traffic. Same attention math, better execution strategy.

**Kernels:** Low-level GPU routines that perform operations like matmul, attention, norm, RoPE, dequantization, and sampling. Better kernels can make the same model much faster.

**Dequantization overhead:** INT4/INT8 weights may need unpacking and scaling during compute. Good runtimes fuse this with matmul; bad ones waste time.

**Tokenizer/sampling:** Usually small compared to model compute, but still part of the pipeline. Sampling converts logits into the next token using temperature, top-k, top-p, etc.

**Thermal throttling:** Laptops may start fast and slow down after heat builds. Sustained inference depends on cooling and power mode, not just specs.

**MoE models:** Total parameters can be huge, but only some experts activate per token. This reduces active compute but creates routing and memory-management complexity.

---



Important nuance: long context hurts **twice**.

```
1. More prompt tokens → slower prefill / time to first token.
2. More cached tokens → larger KV cache and more attention work during decode.
```



```
The model is usually not fully expanded into full FP16 weights permanently.
```

In good runtimes, INT4 weights stay packed/compressed in memory. During computation, kernels often **fuse dequantization with matrix multiplication**. That means the runtime may unpack small blocks, apply scales, multiply, and accumulate, without materializing the entire model as full FP16 in memory.

Wrong: INT4 → fully dequantize entire model → now huge model again

INT4 packed weights sit in memory.
At compute time, blocks are unpacked/scaled as needed.
This saves bandwidth, but adds extra arithmetic and requires optimized kernels.


Also, 4-bit can be slower than 8-bit or 16-bit in some cases because:

```
FP16/BF16 kernels may be extremely optimized on the GPU. INT4 kernels may be more specialized and runtime-dependent.
```

