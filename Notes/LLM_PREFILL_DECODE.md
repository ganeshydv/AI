- **$F_{prefill}$** is the total **work** of the prefill, in FLOPs (floating-point operations). It is how much math the GPU must do to process the whole prompt. It depends only on the model and prompt length, not on the hardware.
- **$T_{prefill}$** is the **time** that work takes, in seconds. It is the time to first token.

$$T_{prefill} = \frac{F_{prefill}}{\text{FLOPS}_{eff}}$$

Work divided by speed gives time, like distance ÷ speed = travel time.

| Symbol | Meaning | Unit | Example (8B, 2k tokens) |
|---|---|---|---|
| $F_{prefill}$ | Total math for the prompt | FLOPs | 33.2 TFLOPs |
| $\text{FLOPS}_{eff}$ | Real speed of the GPU | FLOPs per second | 66 TFLOPS |
| $T_{prefill}$ | Time to process the prompt | seconds | about 0.5 s |

Note that **FLOPs** (lowercase s) is a count of operations, and **FLOPS** (all caps) is operations per second.

The other symbols in $F_{prefill} \approx 2Pn + 2n^2 d_{model} L$:
- $P$: number of parameters (8.03B)
- $n$: number of prompt tokens (2,000)
- $d_{model}$: width of each token vector (4,096)
- $L$: number of layers (32)


Mostly right, with two corrections.

## Inputs you need
- **Weights:** parameter count $P$ (and bytes per weight for memory).
- **Model shape:** $d_{model}$ and layers $L$, from `config.json`. The attention term needs both.
- **Prompt tokens $n$:** the real prompt size you plan to send (instructions + chunks + history + question), not just the context-window maximum.
- **Hardware:** a measured $\text{FLOPS}_{eff}$ and memory bandwidth $BW$.

## Steps
1. **Work:** $F_{prefill} = 2Pn + 2n^2 d_{model} L$ (linear + attention).
2. **Prefill time:** $T_{prefill} = F_{prefill} / \text{FLOPS}_{eff}$.
3. **Decode speed:** $\text{tok/s} \approx BW / (M_{weights} + n \times m_{kv})$.
4. **Total answer time:** $T \approx T_{prefill} + n_{out} / \text{tok/s}$.

## Corrections
- **Time to first token (TTFT) is prefill only** (plus about one decode step). Decode time is not part of it. Decode is the streaming phase after the first token, and its length is $n_{out}$ divided by tok/s, where $n_{out}$ is the answer length.
- **Use your typical prompt size, not only the max.** Max tokens gives the worst case, so calculate both. Also check that memory fits: weights + KV cache for the max context ($n \times m_{kv}$) must fit in VRAM (formulas in 7.7 and section 16).

The result is an estimate. Verify it with `ollama run <model> --verbose`, which reports the actual prompt eval rate and eval rate.


## The formula in simple words

$$F_{prefill} = \underbrace{2Pn}_{\text{linear}} + \underbrace{2n^2 d_{model} L}_{\text{attention}}$$

**Linear part, $2Pn$:** every token goes through every weight once.
- 1 weight × 1 token = 1 multiply + 1 add = 2 FLOPs.
- So 2 × (number of weights) × (number of tokens).
- 8B model, 2,000 tokens: 2 × 8e9 × 2000 = 32 TFLOPs.

**Attention part, $2n^2 d_{model} L$:** every token compares itself with every earlier token.
- That is an $n \times n$ grid of scores. Each score is a dot product over $d_{model}$ numbers (summed across heads), and this is done again for weights × V.
- The causal mask halves it, giving $2 n^2 d_{model}$ per layer, times $L$ layers.
- 2,000 tokens: 2 × 2000² × 4096 × 32 ≈ 1 TFLOP.

The linear part grows with $n$, and the attention part grows with $n^2$. For short prompts the linear part dominates, and for very long prompts attention takes over.

## Your summary, corrected

**Time** and **memory** are two separate calculations. They are not added together.

| Question | Formula | Unit |
|---|---|---|
| How long is prefill? | $\dfrac{2Pn + 2n^2 d_{model} L}{\text{FLOPS}_{eff}}$ | seconds |
| How much memory does the KV cache take? | $2 \times n \times L \times H_{kv} \times d_{head} \times \text{bytes}$ | bytes |

Your KV formula has one fix: use **head size** ($d_{head}$, for example 128), not "dims" ($d_{model}$). With head size, the formula is:

```
2 (K and V) × tokens × layers × K/V heads × head size × bytes
```

8B, 2,000 tokens, fp16: 2 × 2000 × 32 × 8 × 128 × 2 = about 262 MB.

The KV cache does not add to prefill time. It only affects:
- **Memory fit:** weights + KV cache + overhead must fit in VRAM.
- **Decode speed:** the cache must be read for every generated token.


Yes, for Llama 3 8B:

- **4,096 = $d_{model}$:** the width of each token's vector as it travels through the model.
- **128 = $d_{head}$:** the width of Q, K and V inside **one** attention head.

They are linked by the number of query heads:

$$d_{model} = \text{heads} \times d_{head} = 32 \times 128 = 4096$$

The KV cache uses the head size and the **K/V heads** (8, not 32):
- Cached per token per layer: 8 K/V heads × 128 = **1,024** numbers for K, and 1,024 for V.
- It does not use 4,096.

So the full $d_{model}$ is used in the attention FLOPs formula ($2n^2 d_{model} L$, which sums over all 32 heads), and the head size is used in the KV cache formula.

$2\,n^2\,d_{model}\,L$ is the attention math for the whole prompt. Read it as $2 \times n \times n \times d_{model} \times L$.

## Build it up

**1. One score between two tokens.**
- A score is the dot product of one token's Query with another token's Key.
- Over all heads together, that is $d_{model}$ multiplications and $d_{model}$ additions, which is $2\,d_{model}$ FLOPs.

**2. All scores: $n \times n$ of them.**
- Every token is compared with every token, so $n^2$ scores.
- $QK^\top$ therefore costs $2\,n^2\,d_{model}$.

**3. Using the scores: weights × V.**
- Each token blends the Value vectors of the tokens it attends to, which is the same size of work again: $2\,n^2\,d_{model}$.

**4. Add the two products and apply the causal mask.**
- Total: $4\,n^2\,d_{model}$ per layer.
- A token only looks at itself and earlier tokens, so only about half of the $n \times n$ grid is needed.
- Half of $4\,n^2\,d_{model}$ is $2\,n^2\,d_{model}$ per layer.

**5. Repeat for every layer.**
- Multiply by $L$ to get $2\,n^2\,d_{model}\,L$.

## Meaning of each factor

| Factor | Why it is there |
|---|---|
| 2 | Multiply + add, and the two products ($QK^\top$ and weights × V) minus the causal half |
| $n^2$ | Every token against every token |
| $d_{model}$ | Dot-product length summed over all heads (32 × 128) |
| $L$ | Every layer does its own attention |

## Numbers (Llama 3 8B: $d_{model}$ = 4096, $L$ = 32)

| $n$ | Attention FLOPs |
|---|---|
| 2,000 | 2 × 2000² × 4096 × 32 = 1.05 TFLOPs |
| 4,000 | 4.19 TFLOPs (2× tokens gives 4× cost) |
| 8,000 | 16.8 TFLOPs |

The $n^2$ is why long prompts get expensive quickly: doubling the prompt quadruples this part.

Note that the Q, K, V and output projections (Wq, Wk, Wv, Wo) are **not** here. They are weights, so they are already counted in the linear $2Pn$ term. This term covers only the token-to-token comparisons.


### in LLMs in prefill linear part what happens it this part is token id and vector lookup only then vectors goes to layers?

No. The "linear part" is not a pipeline stage. It is the **cost of all the weight-matrix multiplications inside the layers**, added up.

## What happens in prefill (n prompt tokens)

```
1. Token IDs → embedding lookup      table row copy, ~0 FLOPs (not counted in 2Pn)
2. For each of the 32 layers:
     a. Q = X·Wq, K = X·Wk, V = X·Wv      ┐
     b. attention (scores, softmax, ×V)    │ b is the "attention part" (n² term)
     c. output = ·Wo                       ┘
     d. MLP: gate, up, down matrices
3. Final vector of the LAST token → output head → probabilities
```

## What is counted where

| Piece | Weight matrix multiply? | Counted in |
|---|---|---|
| Embedding lookup | No, a table read | Nothing (about 0 FLOPs) |
| Q, K, V projections (Wq, Wk, Wv) | Yes | Linear $2Pn$ |
| Scores $QK^\top$, softmax, weights × V | No weights, token against token | Attention $2n^2 d_{model}L$ |
| Output projection (Wo) | Yes | Linear |
| MLP (gate, up, down) | Yes, the largest share | Linear |
| Output head | Yes, but only for the last token in prefill | Linear (small) |

## Why it is "2 × P × n"

- Each token's vector is multiplied by every weight matrix in all 32 layers, so every weight is used once per token.
- That is 2 FLOPs (multiply + add) × P weights × n tokens.
- All `n` tokens do this at the same time as one big matrix multiply: `[n, 4096] × [4096, 4096]` and so on.

## Rough split of the linear part (Llama 3 8B, per layer)

| Matrices | Parameters | Share |
|---|---|---|
| Attention (Wq, Wk, Wv, Wo) | 42M | about 19% |
| MLP (gate, up, down) | 176M | about 81% |

So most of the linear work is in the MLP. The lookup is the cheap entry step, and the layers are where nearly all the FLOPs happen.

### so why its not layer x P x n x2?

Because $P$ already includes all the layers. It is the total number of weights in the whole model, not the weights of one layer.

For Llama 3 8B:

```
one layer  = 218,112,000 weights
32 layers  = 32 × 218,112,000 = 6.98 billion     ← layers are already inside this
+ embedding table + output head = 8.03 billion   ← this is P
```

The linear cost is therefore:

```
2 FLOPs × (all weights used per token) × n tokens
= 2 × P × n
```

Multiplying by $L$ again would count the layers twice.

## When you would multiply by $L$

| Formula | Uses | Layer count |
|---|---|---|
| $2Pn$ | Total weights $P$ | Already included in $P$ |
| $2 \times P_{layer} \times L \times n$ | Weights of **one** layer | Needed here, since $P_{layer} \times L$ rebuilds $P$ |
| $2n^2 d_{model} L$ (attention) | No weights, only token-against-token math | Needed here, since attention is recomputed in every layer |

Check with numbers: $2 \times 218{,}112{,}000 \times 32 \times n = 2 \times 6.98\text{B} \times n$, which is the same as $2Pn$ (apart from the embedding table and output head).