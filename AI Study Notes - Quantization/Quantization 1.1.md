

Quantization means mapping high-precision continuous-ish weights into fewer representable levels. practical benefit - shrinking models by roughly **2–8x**, reducing memory bandwidth pressure and making large models deployable on smaller hardware.

**Quantization families.** RTN, GPTQ, AWQ, SmoothQuant, FP8, GGUF, EXL2/EXL3.

- key idea: **inference is dominated by repeatedly moving huge tensors through memory and compute units.**
- **Loading path.** Disk → OS page cache → RAM → VRAM.

![[image-8.png]]
![[image-9.png]]


- memory-bandwidth, precision, cache, scheduling, and kernel-execution problem.
- loading/quantization/precision formats like FP32, BF16, INT8, INT4, GGUF, EXL2, AWQ, and the practical problem of fitting models into machine memory.


---------------

- model is weights + tensors + bytes. 
- see an LLM as a giant stack of matrices, not as an abstract “AI brain.”
- **parameters are numbers**, numbers live in **precision formats**, precision determines **bytes**, bytes determine whether the model fits in **RAM/VRAM**, and fitting is the first hard constraint before speed.
- memory equation. `model_memory ≈ num_params * bytes_per_param`
- 7B model is roughly:
	`7e9 * 2 bytes ≈ 14 GB` in FP16/BF16  
	`7e9 * 1 byte ≈ 7 GB` in INT8/FP8-ish storage  
	`7e9 * 0.5 byte ≈ 3.5 GB` in INT4 weight storage, before metadata/scales/overhead

- Real inference also needs metadata, quantization scales, runtime buffers, activations, KV cache, CUDA memory pools, tokenizer objects, OS memory, and framework overhead.

- FP16 has more mantissa precision than BF16 but less exponent range. That means it can represent fine-grained values, but it is more likely to overflow or underflow in some training situations.
- BF16 was designed to preserve the exponent range of FP32 while reducing mantissa precision. 
- BF16 is less precise in the small digits,but better at representing very large/small scale ranges.

------

### Smaller model != always fast load. Even if its smaller by 10-20GB.
- speed depends on the full path:

```
storage format → loading → memory layout → kernel support → dequantization → matmul → cache behavior
```


#### Simplest Quantization: RTN.

- **RTN**, or round-to-nearest. 
`q = round(w / scale) , w_approx = q * scale `

w        = original floating-point weight
scale    = chosen step size
q        = quantized integer code
w_approx = reconstructed approximate weight

- Some weights matter more than others. Some layers are more fragile. Some channels have outliers. Some activations amplify small errors.
- **GPTQ**, **AWQ**, and **SmoothQuant** appear.


-----

Quantization affects at least four things:

```
1. Storage size
2. Runtime memory ( loaded weights occupy less RAM/VRAM. )
3. Memory bandwidth ( huge because autoregressive decoding often becomes bandwidth-limited.)
4. Numerical behavior ( Sometimes quality barely drops. Sometimes reasoning, multilingual ability, coding skill, or instruction following degrades. Quantization error is not uniform across models or tasks. )
```


-------

### Weight-only quantization vs activation quantization

weights      → learned static parameters
activations  → intermediate values produced while processing input

Store weights in low precision.
Keep activations in FP16/BF16 during computation.

common because weights dominate model size, and weight-only quantization is easier to preserve quality with.

Advanced quantization methods are basically ways of asking:

```
Which weights/channels/layers can tolerate error?
Which ones must be preserved carefully?
How do we choose scales so important behavior survives?
```

-----

1. GPTQ is a post-training quantization method that tries to minimize the damage caused by quantizing weights. It uses approximate second-order information to understand which errors matter more.
2. AWQ stands for activation-aware weight quantization.  Some weight channels are more important because activations use them heavily. Protect those important channels more carefully.
3. SmoothQuant is more focused on handling activation outliers. If activations have nasty outliers, SmoothQuant rescales internal channels to reduce activation quantization difficulty while compensating in weights.


### GGUF, EXL2, safetensors: algorithm vs format vs runtime

1. Quantization algorithms
2. File formats
3. Inference runtimes


KV_cache_memory grows with:
number_of_layers
* context_length
* hidden_size / attention structure
* bytes_per_KV_value
* batch_size / number_of_concurrent_sequences


First, the runtime may need to dequantize blocks of weights during computation.
Second, KV cache may still use FP16/BF16 unless separately quantized.
Third, the GPU may not have native INT4 acceleration for that exact layout.

- Neural networks are usually overparameterized. Many weights are not individually sacred. The learned function is distributed across many parameters. Small perturbations to many weights often do not destroy the function.

---------


## Prefill

Prefill takes your input prompt and computes internal representations and KV cache for all prompt tokens.

```
Prompt length = 2000 tokens
```

The model processes those 2000 tokens through all layers. This is expensive, but it has parallelism. The GPU can run large matrix multiplications across many tokens.

Prefill cost grows with prompt length.

```
longer prompt → more prefill work
```

## Decode

Decode generates the answer one token at a time.

```
generate token 1     generate token 2       generate token 3...
```

Each decode step depends on the previous token, so it is sequential.
You cannot fully parallelize future tokens because you do not know them yet.


-------------------------


# Token generation as a pipeline

For each generated token, the runtime roughly does:

```
1. Take current token id
   2. Convert to embedding
      3. Run through layer 1
         4. Update layer 1 KV cache
            5. Run through layer 2
               6. Update layer 2 KV cache...
                  
                  
                  N. Produce logits
                  N+1. Apply sampling rulesN+2. Choose next token
                  N+3. Repeat
```

A conceptual cost model:

```
time_per_token ≈ 
weight_read_time               + matrix_multiply_time               + attention_cache_read_time               + KV_cache_write_time               + CPU/GPU_sync_time               + sampling_time
```

For large models, weight reading and matrix multiplications dominate.

-------------

#### # Why first token latency differs from tokens-per-second

```
time to first token
tokens per second after generation begins
```

first token needs - prompt tokenization, prefill over prompt, KV cache creation, first logits computation. 
model can have decent tokens/sec but slow first-token latency if the prompt is long.

-------------------

- **grouped-query attention** or **multi-query attention** reduces KV cache size by using fewer KV heads than query heads.
- dense model uses most parameters for each token.
- Mixture-of-Experts model has many parameters, but each token activates only some experts.

number of layers
hidden dimension
MLP expansion size
number of attention heads
number of KV heads
vocabulary size
context length
attention type
positional encoding type
MoE vs dense