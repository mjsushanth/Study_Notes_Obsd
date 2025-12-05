
The library is organized around a **core module + pipelines**:
- `dllm/core`: shared bits for **generation**, **schedulers**, and **trainers** (this is where the MDLMTrainer lives, analogous to HF `Trainer`).[GitHub](https://github.com/ZHZisZZ/dllm)
- `dllm/pipelines/{llada,dream,bert,editflow}`: each pipeline packages model definitions, trainer entrypoints, and generator utilities for that family. LLaDA and Dream are canonical DLMs; BERT is “turn any BERT into a diffusion chatbot”; 
- EditFlow adds insertion/deletion/substitution editing.

It ships **batteries-included training**:
- Training scripts (e.g. `examples/llada/sft.py`) take `model_args, data_args, training_args`, build model+tokenizer via `dllm.utils.get_model/get_tokenizer`, then instantiate `dllm.core.trainers.MDLMTrainer` in a very HF-like way.
- It directly supports **LoRA**, **4-bit loading**, and **accelerate configs** (DDP, ZeRO-1/2/3, FSDP) through CLI args like `--load_in_4bit True --lora True` and `--accelerate_config "ddp,zero-2,fsdp"`.
- There are Slurm-ready scripts, but you can also launch locally with `accelerate launch` on a single multi-GPU or single-GPU system.

It also unifies **inference** and **evaluation**:
- Generators such as `LLaDAGenerator` wrap model-specific sampling logic, so your inference code is just: get model+tokenizer, pick the right generator, call `generate()`; the diffusion steps / schedulers are abstracted away.
- For eval, they integrate `lm-evaluation-harness` as a submodule; you can call `dllm/pipelines/llada/eval.py` with tasks like MMLU_Pro via `accelerate launch`, and they provide ready scripts to evaluate LLaDA, Dream, and BERT-chat models across benchmarks.

---

They’ve released **ModernBERT-{base,large}-chat-v0** as BERT-based diffusion chat models, trained via **masked instruction tuning** on public data (TULU-3 SFT mixture + SmolTalk). The dLLM report and blog coverage emphasize that ModernBERT-large-chat reaches about **93% of Llama3-1B’s MMLU performance with ~60% fewer parameters**, highlighting that encoder-only + diffusion can be competitive with small AR LLMs.

### Main insight:

- dLLM is cool: it’s a **trainer+generator ecosystem** that lets you swap in DLM architectures, fine-tune them with HF-style configs, and evaluate them with standard harnesses.
- You don’t have to write a diffusion LM from scratch; you can focus on **data, configs, and analysis**.
- "one orchestrating notebook wrapping a fairly heavy but already-implemented research stack."

---

dLLM lets you transplant that intuition into **discrete diffusion over token sequences**:

- Instead of `x_t` being a noisy pose vector, it’s a masked / corrupted sequence of tokens. The denoiser predicts the _cleaner_ sequence distribution at each step. [AI Engineering+1](https://aiengineering.beehiiv.com/p/build-and-train-diffusion-language-models-from-scratch)
- Bidirectional reasoning and arbitrary-order generation correspond to “you don’t have to march left-to-right; you can revise earlier slots once more context is visible”, which you can directly compare to your AR mental model.
- Nvidia-style TiDAR / DiffuLLaMA work on **bridging diffusion and AR** (e.g., initializing or distilling one into the other, or combining objectives) is conceptually aligned with these experiments: you’re playing in the same design space but at a smaller, controllable scale.[](https://github.com/HKUNLP/DiffuLLaMA?utm_source=chatgpt.com)


- `examples/llada`: LLaDA / LLaDA-MoE – diffusion LMs built on Llama-style decoders.
- `examples/dream`: Dream – another discrete diffusion LM family.
- `examples/bert`: **“BERTs that chat”**, using Masked Diffusion Language Modeling (MDLM).
- `examples/editflow`: EditFlow, which wraps the above with edit operations.

