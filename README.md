# LLM VRAM dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22966137.svg)](https://doi.org/10.5281/zenodo.22966137)

How much GPU memory 29 open-weight language models need, at six quantisations and every common context length, and which of 13 GPUs can run each one.

Architecture values are read from each model's own `config.json` on Hugging Face. GPU memory and bandwidth come from the manufacturers' specification pages. Everything else is computed from those with one stated formula. Nothing is benchmarked, and nothing is copied from another site's table.

It is the data behind the free calculators on [nodegrove.io](https://nodegrove.io), and its home page is [nodegrove.io/data](https://nodegrove.io/data).

Version 2026-09-25 · [CC BY 4.0](LICENSE) · Archived on Zenodo: [10.5281/zenodo.22966137](https://doi.org/10.5281/zenodo.22966137)

## What it shows

- **Newer models pay far less for context.** At 32k tokens, Qwen3 32B (2025, standard attention) spends 8.6 GB on its KV cache; its successor Qwen3.8 27B (2026, hybrid attention) spends 2.3 GB. Across all 29 models, each 1,000 tokens of context costs between 0.006 GB (Nemotron 3.5 Lightning 30B-A3B) and 0.328 GB (DeepSeek-R1 Distill Llama 70B).
- **Active parameters set the speed, not the memory.** Qwen3-Coder-Next 80B-A3B reads 3B parameters per token but keeps all 79.7B in memory: 48.9 GB at Q4_K_M with 8k context, against 22.4 GB for the dense Qwen3 32B.
- **24 of 29 models fit a 24 GB card at Q4_K_M with 8k context.** The largest is Qwen3.6 35B-A3B at 22.4 GB, under the 22.8 GB line (95% of the card).

## Files

| File | Rows | Contents |
|---|--:|---|
| [`llm-vram-models.csv`](llm-vram-models.csv) | 29 | One row per model: size, attention design, context window, licence and the source of every value |
| [`llm-vram-gpus.csv`](llm-vram-gpus.csv) | 13 | One row per GPU: memory, the memory a runtime can use, bandwidth and the specification page |
| [`llm-vram-requirements.csv`](llm-vram-requirements.csv) | 1,068 | Memory per model, quantisation and context length, split into weights, KV cache and overhead |
| [`llm-vram-gpu-fit.csv`](llm-vram-gpu-fit.csv) | 2,262 | Every model on every GPU at every quantisation, at 8k context: verdict, longest context that fits and speed ceiling |
| [`llm-vram.json`](llm-vram.json) |  | Everything above in one file, with the method, its constants and each model's cache layout |

Every file is also served from nodegrove.io with open CORS, so it can be read by URL:

```python
import pandas as pd

fit = pd.read_csv("https://nodegrove.io/data/llm-vram-gpu-fit.csv")
fit.query("gpu_id == 'rtx-4090' and quant == 'Q4_K_M' and verdict != 'no'")
```

## Method

Memory is weights plus KV cache plus overhead.

- **Weights** are parameters × bytes per parameter: FP16 2.00, Q8_0 1.06, Q6_K 0.82, Q5_K_M 0.71, Q4_K_M 0.58, Q3_K_M 0.47. These are effective averages for GGUF files, including the scales and the unquantised embedding and output layers.
- **KV cache** is, for each group of layers, the values cached per token × the tokens held × 2 bytes (1 byte in `total_gb_q8_cache`). A standard transformer caches 2 × KV heads × head dimension on every layer for the whole context; 14 of the 29 models work that way. The other 15 cache less, and are counted as they cache:
  - *sliding-window* (7 models, e.g. Gemma 4 12B, gpt-oss 20B): the windowed layers keep only the last 128 to 1,024 tokens, and the other layers carry the long context.
  - *hybrid* (6, e.g. Qwen3.8 27B, Nemotron 3.5 Lightning 30B-A3B): only the full-attention layers keep a growing cache; the linear-attention or Mamba layers hold a small fixed state.
  - *latent* (2, e.g. Mistral Small 4 119B, GLM-4.7 Flash 30B-A3B): each layer caches one compressed vector per token instead of keys and values for every head.
- **Overhead** is 0.5 GB for the runtime plus 4% of the weights for activations and buffers.
- **Fit.** A model fits a GPU when its total at 8,192 tokens is at most 95% of the memory the runtime can address, and is tight above 85%. Apple silicon gives the GPU about 75% of its unified memory by default, so the Mac Studio M2 Ultra with 192 GB counts as 144 GB.
- **Speed ceiling** is 0.7 × memory bandwidth ÷ bytes of active weights, for one stream generating text.

## Caveats

- These are estimates from published values, not measurements. Real usage moves with the runtime, the batch size, flash attention and cache quantisation. Read "fits" as "worth trying".
- The savings of sliding-window, hybrid and latent attention need a runtime that implements the matching cache. Current llama.cpp and vLLM do for these families; an older build may store every layer at full length and use far more memory at long context.
- The speed figure is a ceiling, and the faster it is, the further real runtimes fall below it: when few weights are read per token, the costs the formula leaves out (kernel launches, expert routing, KV cache reads, sampling) take a larger share of each token.
- Parameter counts are the checkpoint's. Mixture-of-experts models count every expert, because all of them sit in memory. Multimodal checkpoints include the vision encoder, so a text-only GGUF is slightly smaller.
- Context windows are native: `max_position_embeddings`, or the model card where the two differ. No row credits a model with more context than that.

## Columns

Blank means not applicable. The JSON holds the same rows under `models`, `gpus`, `requirements` and `gpu_fit`, plus the method, its constants and each model's cache layout (`kv_groups`).

### llm-vram-models.csv

| Column | Meaning |
|---|---|
| `model_id` | Stable identifier, also the page slug on nodegrove.io |
| `model` | Model name |
| `family` | Model family |
| `released` | Month the weights were published (YYYY-MM) |
| `params_b` | Total parameters, billions, as the checkpoint reports them. Mixture-of-experts models count every expert; multimodal checkpoints include the vision encoder |
| `active_params_b` | Parameters read per token, billions. Mixture-of-experts models only; blank for dense models |
| `attention` | How the model caches context: standard, sliding-window, hybrid or latent (see the method) |
| `layers` | Transformer layers (blocks) |
| `kv_heads` | Key/value heads of the main attention layers |
| `head_dim` | Dimension of each key/value head |
| `full_cache_layers` | Layers whose cache grows with the whole context |
| `window_layers` | Layers that only keep the last window_tokens tokens |
| `window_tokens` | Sliding-window length in tokens; blank if the model has none |
| `kv_cache_gb_per_1k_tokens` | Memory each extra 1,000 tokens of context costs, GB, with an FP16 cache once any windows are full |
| `fixed_state_gb` | Fixed recurrent state of linear-attention or Mamba layers, GB. Does not grow with context |
| `context_tokens` | Native context window in tokens (max_position_embeddings, or the model card where the two differ) |
| `license` | Licence of the weights, from the model card |
| `license_permissive` | true when the licence has no field-of-use or scale conditions |
| `role` | code or reasoning for specialist models; blank for general-purpose models |
| `successor_id` | Newer model in the same family and role, if there is one |
| `hf_repo` | Hugging Face repository the values were read from |
| `hf_gated` | true when config.json opens only after accepting the licence on Hugging Face |
| `config_url` | Link to the config.json the architecture values come from |
| `page_url` | The model page on nodegrove.io |
| `attention_detail` | The attention layout in one sentence |
| `notes` | Caveats that change how the numbers should be read |

### llm-vram-gpus.csv

| Column | Meaning |
|---|---|
| `gpu_id` | Stable identifier, also the page slug on nodegrove.io |
| `gpu` | Card or machine |
| `kind` | consumer, workstation, datacenter or apple |
| `memory_gb` | Memory from the manufacturer specification, GB (unified memory for Apple silicon) |
| `usable_gb` | Memory an inference runtime can address, GB. Apple silicon gives the GPU about 75% of unified memory by default |
| `budget_gb` | 95% of usable_gb: the line a model must fit under to count as fitting |
| `bandwidth_gb_s` | Memory bandwidth from the manufacturer specification, GB/s |
| `spec_url` | The manufacturer's specification page |
| `page_url` | The GPU page on nodegrove.io |
| `notes` | Caveats |

### llm-vram-requirements.csv

| Column | Meaning |
|---|---|
| `model_id` | Joins llm-vram-models.csv |
| `model` | Model name |
| `quant` | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M or Q3_K_M |
| `bytes_per_param` | Effective bytes per parameter at that quantisation |
| `context_tokens` | Tokens held in context |
| `weights_gb` | Parameters × bytes_per_param, GB |
| `kv_cache_gb` | KV cache stored at FP16, plus any fixed state, GB |
| `overhead_gb` | Runtime context and buffers: 0.5 GB + 4% of weights |
| `total_gb` | weights_gb + kv_cache_gb + overhead_gb |
| `total_gb_q8_cache` | The same total with the KV cache quantised to 8 bits |

### llm-vram-gpu-fit.csv

| Column | Meaning |
|---|---|
| `model_id` | Joins llm-vram-models.csv |
| `model` | Model name |
| `gpu_id` | Joins llm-vram-gpus.csv |
| `gpu` | Card or machine |
| `quant` | Quantisation of the weights |
| `context_tokens` | Context the fit is computed at (8,192 tokens) |
| `need_gb` | Estimated memory at that quantisation and context, GB (total_gb in llm-vram-requirements.csv) |
| `verdict` | fits: need_gb is within the GPU's budget_gb. tight: it fits, but above 85% of usable memory. no: it does not fit |
| `max_context_tokens` | Longest context that fits at this quantisation with an FP16 cache, capped at 131,072 tokens or the model's window. 0 when the weights alone do not fit |
| `tokens_per_second_ceiling` | Upper bound on single-stream generation speed, tokens/s (see the method). Blank when the model cannot load |

## Versions

Each data version is a dated [release](https://github.com/nodegrove/llm-vram-dataset/releases), archived on Zenodo with a DOI of its own; [10.5281/zenodo.22966137](https://doi.org/10.5281/zenodo.22966137) covers all versions and resolves to the newest. New open models are added as they come out. If a value disagrees with its source, [open an issue](https://github.com/nodegrove/llm-vram-dataset/issues) with the row and the link, and the next version will carry the fix.

## Licence and citation

[CC BY 4.0](LICENSE). Use it for anything, commercial work included, with credit to **Nodegrove** and a link to https://nodegrove.io/data.

```text
Nodegrove (2026). LLM VRAM dataset (version 2026-09-25) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22966137
```

Each model's weights carry their own licence, listed per row. This repository contains no model files. Model and GPU names are trademarks of their owners.

---

Made by [Nodegrove](https://nodegrove.io), a private AI server in the cloud. The same numbers power its free tools: [Can I run it?](https://nodegrove.io/tools/can-i-run-it) · [LLM VRAM calculator](https://nodegrove.io/tools/llm-vram-calculator) · [Speed estimator](https://nodegrove.io/tools/llm-speed-estimator) · [Every model compared](https://nodegrove.io/models)
