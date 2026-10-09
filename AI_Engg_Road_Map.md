1. LLM Fundamentals
Start by getting these concepts really strong:
Tokenization—converting text into tokens and token IDs
Embeddings — representing tokens as vectors
Attention / Self-Attention
Transformer Architecture
Positional Encoding / RoPE
Encoder vs Decoder vs Encoder-Decoder
Autoregressive Generation
Context Window
Logits, Softmax, Probability
Temperature, Top-k, Top-p
Parameters vs Tokens
 2. LLM Training
You should understand the basic LLM training pipeline:
Pretraining
Next-token prediction
Cross-entropy loss
Gradient descent / Backpropagation
Batch size & Learning rate
Adam / AdamW
Learning-rate scheduling
Mixed precision
Distributed training
Data parallelism / Model parallelism
Checkpointing
A useful distinction:
Pretraining = teaching the model language and general patterns
 Post-training = teaching the model desired behavior
3. Post-Training & Alignment
This is how a pretrained LLM becomes more useful as an assistant:
SFT — Supervised Fine-Tuning
Instruction tuning
RLHF
Reward models
DPO
Preference optimization
Alignment
Safety tuning
4. Fine-Tuning
Very important for AI Engineers:
Full fine-tuning
LoRA
QLoRA
PEFT
Adapter tuning
Quantization
4-bit / 8-bit models
Fine-tuning dataset preparation
Evaluating models before and after fine-tuning
5. RAG — Retrieval-Augmented Generation
One of the most important concepts for real-world LLM applications:
User Query → Embedding → Retrieval → Relevant Context → LLM → Answer
Learn:
Chunking
Embeddings
Vector databases
Similarity search
Cosine similarity
Metadata filtering
Hybrid search
BM25
Reranking
Context injection
Query rewriting
Multi-query retrieval
RAG evaluation
Naive RAG vs Advanced RAG
6. AI Agents
Understand how an LLM can become a decision-making system:
Tool calling
Function calling
Planning
ReAct
Agent loops
Memory
State management
Multi-agent systems
MCP
Human-in-the-loop
Agent evaluation
Guardrails
An important mental model:
LLM ≠ Agent
 LLM + Tools + State + Control Loop ≈ Agent
7. Inference & Serving
This is where AI Engineering starts becoming serious systems engineering:
KV Cache
Prefill vs Decode
Batching
Continuous batching
Streaming
Latency
Throughput
Tokens/sec
TTFT — Time to First Token
TPOT — Time Per Output Token
Quantization
Speculative decoding
Model serving
GPU memory management
vLLM and other inference engines
8. LLM Evaluation
“Looks good to me” is not an evaluation strategy 😄
Learn:
Exact Match
Precision / Recall / F1
BLEU / ROUGE
LLM-as-a-Judge
Human evaluation
Groundedness
Faithfulness
Relevance
Toxicity / Safety evaluation
Hallucination evaluation
RAG evaluation
Regression testing
 9. LLM Security
This is often overlooked but extremely important:
Prompt injection
Indirect prompt injection
Jailbreaking
Data leakage
Sensitive information exposure
Insecure tool calling
Excessive agent permissions
RAG poisoning
Model supply-chain risks
Output validation
Guardrails
10. Production / LLMOps
Building a prototype is easy. Building a reliable, scalable, and cost-efficient production system is AI Engineering.
Important concepts:
Prompt versioning
Model versioning
Observability
Tracing
Logging
Cost tracking
Latency monitoring
Rate limiting
Caching
Retries
Fallback models
A/B testing
Evaluation pipelines
Data pipelines
CI/CD for AI systems

Recommended Learning Order
If I were learning this from scratch, I'd follow this sequence:
Phase 1 — Foundations
Python → PyTorch → Neural Networks → Transformers → Attention
Phase 2 — LLMs
Tokenization → Embeddings → GPT architecture → Sampling → Context windows
Phase 3 — Adaptation
Prompting → SFT → LoRA/QLoRA → DPO
Phase 4 — Applications
RAG → Vector DB → Reranking → Tool Calling → Agents
Phase 5 — Production
Evaluation → LLMOps → Inference → vLLM → Quantization → Observability
Phase 6 — Advanced
Distributed training → MoE → Speculative decoding → Long-context techniques → Multimodal LLMs
