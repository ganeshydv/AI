# LLM Prefill and Decode: Calculation Notes

_Goal: estimate how long an LLM takes to answer and whether it fits in GPU memory. Example model: Llama 3 8B._

## 1. The two phases

| | Prefill | Decode |
|---|---|---|
| What | Process the whole prompt (`n` tokens) in parallel | Generate the answer, 1 token at a time |
| Limited by | **Compute** (FLOPS) | **Memory bandwidth** (bytes read per token) |
| You feel it as | Time to first token (TTFT) | Tokens per second |

**TTFT = prefill only.** Decode is the streaming that follows.

## 2. Symbols

| Symbol | Meaning | Llama 3 8B |
|---|---|---|
| $P$ | Parameters (weights), whole model, all layers included | 8.03B |
| $n$ | Prompt tokens (instructions + chunks + history + question) | e.g. 2,000 |
| $d_{model}$ | Width of a token vector | 4,096 |
| $L$ | Layers | 32 |
| $d_{head}$ | Width of Q, K, V in one head | 128 |
| $H_{kv}$ | K/V heads | 8 (32 query heads) |

- $d_{model} = \text{heads} \times d_{head} = 32 \times 128$.
- **FLOPs** (lowercase s) = a count of operations (work). **FLOPS** = operations per second (speed).

## 3. Prefill work: $F_{prefill}$

$$F_{prefill} = \underbrace{2Pn}_{\text{linear}} + \underbrace{2n^2 d_{model} L}_{\text{attention}}$$

### Linear part, $2Pn$
- Every token goes through every weight once. One weight = multiply + add = **2 FLOPs**.
- So 2 × weights × tokens. It is the cost of all weight multiplies inside the layers (Wq, Wk, Wv, Wo and the MLP), not a separate stage.
- **$P$ already contains all layers.** Do not multiply by $L$ again. (Only if you know one layer's weights: $2 \times P_{layer} \times L \times n$.)
- The embedding lookup is a table read (about 0 FLOPs). The output head runs only for the last prompt token. So the effective $P$ is about 7.0B, and the linear part is about 13% lower than the formula.
- Layer split: MLP about 81% of the weights, attention projections about 19%.

### Attention part, $2n^2 d_{model} L$
Token-against-token math. No weights here (the projections are in the linear part).

1. One score = dot product over $d_{model}$ numbers (all heads together) = $2\,d_{model}$ FLOPs.
2. $n \times n$ scores, so $QK^\top$ costs $2n^2 d_{model}$.
3. weights × V costs the same, so together $4n^2 d_{model}$.
4. The causal mask (a token only looks backwards) halves it: $2n^2 d_{model}$ per layer.
5. Every layer repeats it: × $L$.

Linear grows with $n$, attention grows with $n^2$. Short prompts: linear dominates. Very long prompts: attention takes over.

### Worked example, $n$ = 2,000

```
linear    = 2 × 8.03e9 × 2000        = 32.1 TFLOPs
attention = 2 × 2000² × 4096 × 32    =  1.05 TFLOPs
F_prefill                            ≈ 33.2 TFLOPs
```

| $n$ | Linear | Attention | Total |
|---|---|---|---|
| 2,000 | 32.1 | 1.05 | 33.2 TFLOPs |
| 4,000 | 64.2 | 4.19 | 68.4 TFLOPs |
| 8,000 | 128.5 | 16.8 | 145.3 TFLOPs |

Doubling the prompt doubles the linear part but quadruples the attention part.

## 4. Prefill time: $T_{prefill}$

$$T_{prefill} = \frac{F_{prefill}}{\text{FLOPS}_{eff}}$$

Work ÷ speed = time (like distance ÷ speed).

**Where GPU FLOPS comes from:**

$$\text{FLOPS}_{peak} = \text{cores} \times \text{clock} \times \text{FLOPs per cycle (2: one fused multiply-add)}$$

RTX 4090: 16,384 × 2.52 GHz × 2 ≈ 82.6 TFLOPS (CUDA cores). LLM math runs on tensor cores, about 165 TFLOPS fp16.

**Peak is a fixed ceiling, effective is lower and varies** (often 30 to 70% of peak):
- number format (use the one the math really runs in: fp16, even with 4-bit weights),
- clocks and power limits,
- kernel quality and prompt size (short prompts underuse the GPU),
- memory stalls.

**Measure it:** `ollama run <model> --verbose`, read `prompt eval rate` (tok/s). Then $\text{FLOPS}_{eff} \approx 2P \times \text{rate}$.

| $n$ | Total work | At 5 TFLOPS | At 66 TFLOPS |
|---|---|---|---|
| 2,000 | 33.2 TFLOPs | about 6.6 s | about 0.5 s |
| 4,000 | 68.4 TFLOPs | about 13.7 s | about 1.0 s |
| 8,000 | 145.3 TFLOPs | about 29 s | about 2.2 s |

**Why prefill is compute-bound:** the `n` tokens are multiplied as one big matrix (`[2000, 4096] × [4096, 4096]`), so each weight read from VRAM is reused for all 2,000 tokens. The cores have plenty of math per byte. Quantization barely helps here (math is still fp16).

**Cached prompt:** if the runtime reuses the KV cache of an unchanged prompt start, only the $m$ new tokens are prefilled. Linear becomes $2Pm$, and attention about $2\,m\,n\,d_{model} L$.

## 5. KV cache: memory, not time

$$M_{kv} = 2 \times n \times L \times H_{kv} \times d_{head} \times \text{bytes}$$

- 2 = one K and one V. It uses **head size** (128) and **K/V heads** (8), not $d_{model}$ (4,096).
- Llama 3 8B fp16: 2 × 32 × 8 × 128 × 2 = 128 KB per token, so 2,000 tokens = about 262 MB.
- It does **not** add to prefill time. It affects:
  - **fit:** weights + KV cache + overhead must fit in VRAM,
  - **decode speed:** the cache is read for every generated token.

## 6. Decode

To produce one token the GPU reads all weights and the whole KV cache from VRAM:

$$\text{tok/s} \approx \frac{BW}{M_{weights} + M_{kv}}$$

- $M_{kv} = n \times m_{kv}$ is the **total** KV cache (section 5). $m_{kv}$ is the cache per token (128 KB for Llama 3 8B fp16).
- So the bytes read per token = all weights + the whole KV cache.
- Weights are loaded into VRAM **once** and stay there. Each token only **reads** them (a memory read, not a disk load).
- The on-chip cache is only tens of MB, so weights can't be kept in the cores. They are streamed every step.
- Example: 8B fp16 (16 GB) on a 1,000 GB/s GPU: about 1000 / 16.3 ≈ 60 tok/s upper bound. 4-bit (about 5 GB) is about 3× faster.
- Exceptions to "all weights per token": batching (one read serves many requests), speculative decoding (one read verifies several tokens), MoE models (only active experts are read).

## 7. Full recipe for any model

1. **Inputs:** $P$, $d_{model}$, $L$, $H_{kv}$, $d_{head}$ (from `config.json`), bytes per weight, typical and max $n$, measured $\text{FLOPS}_{eff}$ and $BW$.
2. **Work:** $F_{prefill} = 2Pn + 2n^2 d_{model} L$.
3. **TTFT:** $T_{prefill} = F_{prefill} / \text{FLOPS}_{eff}$.
4. **Memory fit (separate check):** $M_w + n \times m_{kv} + M_{overhead} \le \text{VRAM}$.
5. **Decode speed:** $\text{tok/s} \approx BW / (M_w + M_{kv})$.
6. **Total time:** $T_{prefill} + n_{out} / \text{tok/s}$ ($n_{out}$ = answer length).
7. Do it for the typical prompt **and** the maximum. Verify with `ollama run <model> --verbose`.

Results are estimates. Real speeds are below peak.

## 8. Common mix-ups

| Mix-up | Correct |
|---|---|
| Linear = layers × P × n × 2 | $P$ already has all layers: $2Pn$. Only attention is × $L$ |
| $d_{model}$ and $d_{head}$ are the same | $d_{model}$ = heads × $d_{head}$. Attention FLOPs use $d_{model}$, KV cache uses $d_{head}$ |
| "Linear part" is the embedding lookup | It is all the weight multiplies in the layers. The lookup is about 0 FLOPs |
| A GPU has one constant FLOPS | Peak is a ceiling. Effective varies |
| Decode time is part of TTFT | TTFT = prefill only |
| KV cache adds to prefill time | It is memory. It limits fit and slows decode |
| Weights are reloaded for each token | Loaded once. Each token reads them from VRAM |

## 9. Formula sheet

| What | Formula |
|---|---|
| Prefill work | $2Pn + 2n^2 d_{model} L$ |
| Prefill time (TTFT) | work ÷ $\text{FLOPS}_{eff}$ |
| Peak FLOPS | cores × clock × 2 |
| KV cache | $2 \times n \times L \times H_{kv} \times d_{head} \times \text{bytes}$ |
| Decode speed | $BW / (M_w + M_{kv})$ |
| Total answer time | $T_{prefill} + n_{out} / \text{tok/s}$ |
