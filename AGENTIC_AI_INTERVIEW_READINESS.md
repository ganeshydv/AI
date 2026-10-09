# Complete Interview Prep Guide — Agentic AI Engineer + Backend Developer

_One knowledge base to prep for **two overlapping tracks**: (1) Agentic AI Engineer / AI Integration Engineer, and (2) Backend Developer / Software Engineer (Node.js + Java/Spring background). Built from everything already in this workspace — the 6 AI docs in this folder, `_JAVA/`, `_DSA_NeetCode/`, `1_OS/`, `2_Networking/`, `4_DataBase_RDS_DynamoDB/` — plus externally-verified LLD/HLD/DSA references (`system-design-primer`, `awesome-low-level-design`)._

---

## 0. How to use this guide

| Track | Weight these sections most |
|---|---|
| **Agentic AI Engineer / AI Integration Engineer** | §3E (Agentic/LLM), §3F (Security), §4 rounds A/B/C, plus HLD §3C for scaling AI systems |
| **Backend Developer / SDE** | §3A (DSA), §3B (LLD), §3C (HLD), §3D (Backend fundamentals), §4 rounds D/E/F |
| **Either — you're a hybrid candidate** | All of it. Most companies hiring "AI Integration Engineer" today still run a standard SDE loop (DSA + LLD + HLD) plus an AI-specific deep-dive round — treat this as the default assumption. |

---

## 1. Where you actually stand (evidence, not guesses)

### AI / Agentic systems (FlowForge/AiHelper project)

| Area | Evidence in your docs |
|---|---|
| RAG pipeline end-to-end | Chunking, embeddings (Xenova MiniLM-384), Qdrant cosine search, prompt assembly |
| Agentic tool-calling loop | `agentLoop.js` (Ollama) + `copilotAgent.js` (Copilot SDK), ReAct-style loop with `MAX_ITERATIONS` guard |
| Tool protocol | Hand-built MCP server, two-phase plan/execute safety pattern |
| Security thinking | Prompt-injection mitigation, credential handling, authZ gap analysis, least-privilege tool allow-listing |
| Async/streaming internals | SSE token streaming, async generators, event loop, process pools for CPU-bound embedding |
| Systems design at scale | Tier 0→5 design (10K→1B users): sharding, GPU embedding services, multi-region, LLM routing |

### Backend fundamentals (prior, verified from your own notes)

| Area | Evidence |
|---|---|
| Java / Spring Boot | `_JAVA/SpringBoot/` — request handling, validation, exception handling, dependency resolution, JPA entity/HQL relations |
| ORM/JDBC/Hibernate | `_JAVA/2_JDBC_.md`, `3_Hibernate_.md`, `3.1.x_JPA_.md` — mapping, architecture |
| Databases | `4_DataBase_RDS_DynamoDB/` — SQL fundamentals, schema design, entity relations, RDS vs DynamoDB trade-offs |
| Operating Systems | `1_OS/` — process/thread, scheduling, context switching, stack vs heap, system calls, **and your own note on how Node handles millions of requests on one thread** (directly relevant to both backend and AI-streaming interviews) |
| Networking | `2_Networking/` — DNS, TLS handshake, TCP/IP, ARP/NAT/DHCP, proxy/reverse proxy — deep enough for HLD networking questions |
| Cloud/AWS | `0_AWS*/` folders — Lambda, DynamoDB, SAM, serverless patterns |
| DSA practice | `_DSA_NeetCode/DsaSheet/` — Array, Binary Search, Hash, Kadane's/Sliding Window, Two Pointers, plus separate LinkedList/recursion/sorting folders |

**Read across both tables: you're not "an AI person learning backend" or "a backend person learning AI" — you have real evidence in both, which is a genuinely rare combination in the current job market.** The gap isn't raw exposure — it's (a) closing specific pattern/topic holes (identified below), and (b) drilling recall speed and articulation under interview time pressure.

---

## 2. Interview-ready project narratives (use these as base stories)

Have a 60-second and a 5-minute version of **both** ready — pick whichever fits the JD.

### Narrative 1 — AI project (STAR)

- **Situation:** Built an AI assistant that answers questions over live platform metadata and can take real actions (create pipelines/connections) via chat.
- **Task:** Needed retrieval that's accurate over structured data, an agent that can safely call tools, and a design that survives a prompt-injection or credential-leak attempt.
- **Action (pick 2-3 to go deep on depending on the role):**
  - RAG: turned Postgres rows into natural-language "documents", embedded with a local ONNX model, stored in Qdrant with deterministic UUIDv5 point IDs for idempotent re-indexing.
  - Agent loop: ReAct-style loop bounded by `MAX_ITERATIONS`, tool allow-listing, and a two-phase plan→confirm→execute pattern so the LLM can *preview* a write but never execute one directly.
  - Security: explicitly separated the read/RAG path (never sees tools) from the action/MCP path (never sees raw retrieved text) so a poisoned document can't trigger a write.
  - Scaling: designed the growth path from 1 Qdrant collection per user (breaks at ~100K users) to a shared collection with payload-based filtering, then sharding by `hash(userId)`.
- **Result:** Working system with an explicit, documented threat model and a scaling plan — not just a demo.

### Narrative 2 — Backend narrative (template, plug in your real Spring/Node project specifics)

- **Situation:** [A Spring Boot / Node service you built or maintained — pick your strongest one]
- **Task:** [The concrete problem — e.g., request validation gaps, N+1 queries, slow endpoint, exception handling inconsistency]
- **Action:** Tie directly to what you have notes on — request validation (`2.1_spring_boot_request_validation_.md`), exception handling (`3_spring_boot_exception_handling_.md`), entity/HQL relations (`5.2_spring_boot_hql_relations.md`), JDBC/Hibernate mapping choices.
- **Result:** [Quantify if possible — latency reduced, bug class eliminated, etc.]

Practice both out loud without reading them. Interviewers probe wherever you slow down — that tells you what to study next.

---

## 3. Master knowledge checklist (by domain)

Legend: ✅ = you have real evidence for this · 🟡 = partial/needs refresh · ⬜ = verify or build

### A. DSA (Data Structures & Algorithms)

Your `_DSA_NeetCode/DsaSheet/` already covers: Arrays, Binary Search, Hashing, Kadane's/Sliding Window, Two Pointers (plus separate LinkedList, recursion, sorting folders). Standard interview pattern coverage:

| Pattern | Status | Notes |
|---|---|---|
| Arrays / Two Pointers / Sliding Window | ✅ | `DsaSheet/Array`, `TwoPointers`, `Kadanes_Algo_Sliding_Window` |
| Hashing | ✅ | `DsaSheet/Hash` |
| Binary Search (incl. on answer) | ✅ | `DsaSheet/binarySearch` |
| Linked Lists | ✅ | `LinkedListEx/` |
| Recursion / Backtracking | 🟡 | `recurssion/` exists — confirm backtracking (subsets/permutations/N-Queens) is included, not just plain recursion |
| Sorting (+ when to use which) | ✅ | `sorting/` |
| Subsets / Combinations | ✅ | `DsaSheet/Subset` |
| Trees (BST, traversals, balanced trees) | ⬜ | Not visible — verify/add |
| Graphs (BFS/DFS, topological sort, union-find, shortest path) | ⬜ | Not visible — verify/add, common at mid+ level |
| Dynamic Programming (1D/2D, knapsack, LCS/LIS patterns) | ⬜ | Not visible — highest-leverage gap, very commonly asked |
| Heaps / Priority Queue (top-K, merge-K) | 🟡 | `5_top_k_frequent_ele.java` touches this — expand to heap-based approach specifically |
| Tries | ⬜ | Not visible — needed for word-search/autocomplete style questions |
| Monotonic Stack/Queue | ⬜ | Not visible — common for "next greater element" class (you have `MaxNext/` — check if this covers it) |
| Bit manipulation | ⬜ | Not visible — usually a handful of quick-win questions |
| Intervals (merge/insert) | ⬜ | Not visible |
| Greedy | ⬜ | Not visible |

**Practice sets to use (external, verified):** NeetCode 150 / Blind 75 for pattern coverage; keep solving in your existing `_DSA_NeetCode` structure so notes stay centralized.

### B. LLD / OOD (Low-Level Design & Object-Oriented Design)

| Topic | Status |
|---|---|
| OOP fundamentals (classes, interfaces, encapsulation, abstraction, inheritance, polymorphism) | ✅ (Java background) |
| Class relationships (association, aggregation, composition, dependency) | 🟡 — know informally, practice naming them precisely |
| Design principles (SOLID, DRY, YAGNI, KISS) | 🟡 — likely used intuitively; drill naming + textbook definitions |
| Design patterns — creational (Singleton, Factory, Abstract Factory, Builder, Prototype) | 🟡 |
| Design patterns — structural (Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy) | 🟡 |
| Design patterns — behavioral (Strategy, Observer, Command, State, Template Method, Visitor, Mediator, Memento, Chain of Responsibility, Iterator) | 🟡 |
| UML (class/sequence/activity/state diagrams) | 🟡 — you already draw mermaid sequence/flow diagrams in your AI docs, same skill |
| Concurrency for LLD rounds (mutex, semaphore, deadlock/livelock, thread pool, producer-consumer, reader-writer) | ✅ (OS notes) + 🟡 (apply to Java-specific concurrency primitives) |
| Common LLD interview problems | ⬜ practice — Parking Lot, LRU Cache, Rate Limiter, Elevator System, Splitwise, Tic-Tac-Toe, Chess, Movie Ticket Booking, Vending Machine |

**How to answer any LLD problem (repeatable process):** clarify requirements → identify core entities/nouns → define relationships → apply SOLID → sketch class diagram → handle edge cases/concurrency → write clean interfaces before implementation.

### C. HLD / System Design

You've already done a full tier-by-tier design exercise (`SCALING_HLD.md`) — that's more hands-on system design practice than most candidates bring. Map it against the standard topic list so nothing's missing:

| Topic | Status |
|---|---|
| Scalability vs performance, latency vs throughput | ✅ (implicit in `SCALING_HLD.md`) |
| CAP theorem, consistency patterns (weak/eventual/strong) | 🟡 — know the practice, drill the vocabulary |
| Availability patterns (failover, replication, 9s math) | 🟡 |
| DNS, CDN (push vs pull) | ✅ (`2_Networking/0_dns__.md`) |
| Load balancers (L4 vs L7), reverse proxy | ✅ (`2_Networking/0_Proxy__reverseProxy__.md`, `___nginx_load_balancer__.txt`) |
| Microservices, service discovery | 🟡 — designed in `SCALING_HLD.md` Tier 3, not deployed |
| DB scaling: replication (master-slave/master-master), federation, sharding, denormalization, SQL tuning/indexing | ✅ designed in `SCALING_HLD.md`; ✅ SQL fundamentals in `4_DataBase_RDS_DynamoDB/` |
| NoSQL types (key-value, document, wide-column, graph) + SQL vs NoSQL decision | ✅ (RDS vs DynamoDB notes) |
| Caching (client/CDN/web/app/DB level, cache-aside/write-through/write-behind/refresh-ahead) | 🟡 designed (`IMPROVEMENT_GUIDE.md` §10), not implemented |
| Asynchronism (message/task queues, backpressure) | 🟡 designed (Kafka/SQS at Tier 3), not built |
| Communication protocols (HTTP, TCP vs UDP, RPC, REST) | ✅ (networking notes + REST APIs in your projects) |
| Security basics for HLD rounds (encrypt in transit/rest, sanitize input, least privilege) | ✅ (demonstrated in MCP security work) |
| Back-of-envelope estimation (latency numbers, powers of two, QPS/storage math) | ⬜ — practice this explicitly, it's a distinct skill from knowing the concepts |
| Classic design problems | ⬜ practice — URL shortener, Twitter feed, chat system (WhatsApp), rate limiter, web crawler, key-value store, notification system |

**Your strongest asset:** you can walk an interviewer through a *real* tier-by-tier scaling design (`SCALING_HLD.md`) instead of reciting memorized theory — lead with that, then generalize to whatever problem they give you.

### D. Backend Fundamentals (DB, query optimization, APIs, streaming)

| Topic | Status |
|---|---|
| Indexing (B-tree, composite, covering indexes, when an index *hurts* writes) | 🟡 — you have SQL notes; drill the query-planner/EXPLAIN side specifically |
| Normalization vs denormalization | ✅ (`0_Table_Schema_DB.md`, `EntityRelations.md`) |
| ACID, transactions, isolation levels (read uncommitted/committed, repeatable read, serializable) + anomalies (dirty/non-repeatable/phantom reads) | 🟡 — verify you can name all 4 levels and their trade-offs cold |
| Query optimization (N+1 problem, EXPLAIN plans, connection pooling) | 🟡 — Hibernate lazy/eager loading notes exist; connect this explicitly to N+1 |
| ORM pitfalls (lazy vs eager loading, first/second-level cache, HQL) | ✅ (`3_Hibernate_.md`, `3.1.2_Hibernate_JPA_Mapping_.md`, `5.2_spring_boot_hql_relations.md`) |
| RDS vs DynamoDB / SQL vs NoSQL decision | ✅ (`RDS_VS_DynamoDB_.md`) |
| REST API design (versioning, idempotency, pagination, status codes, rate limiting) | 🟡 |
| Streaming/real-time (SSE vs WebSockets vs long polling, backpressure) | ✅ (`ASYNC_STREAMING_GUIDE.md` — genuinely strong here) |
| Message queues / event-driven basics (Kafka, SQS, pub/sub) | 🟡 designed at Tier 3 in `SCALING_HLD.md`, not hands-on |
| Concurrency & OS (process vs thread, scheduling, context switching, single-threaded event loop model) | ✅ (`1_OS/`, plus your own "how Node handles millions of requests on one thread" note) |
| Networking fundamentals (TCP/UDP, TLS handshake, DNS resolution path) | ✅ (`2_Networking/`) |
| Docker / containers | 🟡 (`7_Docker/` exists — confirm depth: multi-stage builds, networking, volumes) |

### E. Agentic AI / LLM Engineering

Fully detailed in [AGENTIC_AI_ROADMAP.md](./AGENTIC_AI_ROADMAP.md) (14 sections) — condensed pointer here so this doc stays the single interview-facing index:

| Sub-area | Status | Where to go deep |
|---|---|---|
| LLM/transformer internals (tokenization, attention, sampling, training stages) | ⬜ biggest theory gap | `AGENTIC_AI_ROADMAP.md` §1 |
| Agent loop patterns (ReAct/Plan-and-Execute/Reflexion) | ✅/⬜ mixed | `AGENTIC_AI_ROADMAP.md` §3 |
| Tool use / MCP / function calling | ✅ | `AGENTIC_AI_ROADMAP.md` §4 |
| Memory systems | ✅/⬜ mixed | `AGENTIC_AI_ROADMAP.md` §5 |
| Multi-agent systems | ⬜ biggest practical gap | `AGENTIC_AI_ROADMAP.md` §7 |
| Evaluation/observability | 🟡 | `AGENTIC_AI_ROADMAP.md` §8 |
| Retrieval/RAG depth | ✅/🟡 mixed | `AGENTIC_AI_ROADMAP.md` §10, `Notes/RAG_COMPLETE_GUIDE.md` |
| Frameworks (LangGraph/CrewAI/AutoGen/OpenAI Agents SDK) | ⬜ | `AGENTIC_AI_ROADMAP.md` §11 |

### F. Security (cross-cutting, both tracks)

| Topic | Status |
|---|---|
| OWASP Top 10 awareness (injection, broken authN/authZ, SSRF, etc.) | 🟡 — you've applied specific instances (prompt injection, SQL SELECT-only guard) but haven't necessarily mapped to the full OWASP list |
| SQL injection prevention (parameterized queries) | ✅ (implicit in your backend work) |
| AuthN/AuthZ (JWT/OAuth, session vs token) | 🟡 — documented as a known gap in your own project (`AI_ASSISTANT_DEEP_DIVE.md` §7b) |
| Secrets handling (never round-tripping through logs/LLM context) | ✅ |
| Principle of least privilege (tool allow-listing, scoped credentials) | ✅ |
| Prompt injection (AI-specific) | ✅ — a differentiator few backend-only or AI-only candidates can speak to |

### G. Behavioral & Resume

- Prepare 4-6 STAR stories covering: a conflict/disagreement, a failure and recovery, a time you drove a design decision, a time you had to learn something fast, a time you mentored/reviewed someone's work, a scaling/performance win.
- Resume bullets should follow: **Action verb + what you built + technical specifics + measurable/qualitative result.** E.g., "Designed a two-phase plan/execute safety pattern for an LLM tool-calling agent, preventing unreviewed writes to production metadata."

---

## 4. Interview question bank (organized by round type)

### A. LLM/AI fundamentals (screening rounds)
- What does an embedding actually represent, and why does dimensionality matter?
- Walk through what happens between "user sends a prompt" and "first token appears."
- Why does RAG reduce hallucination, and what does it *not* fix?
- Cosine vs dot-product vs Euclidean distance — when does each matter?
- What's the difference between fine-tuning and RAG, and how do you decide?

### B. Agentic system design (deep-dive round on your project)
- How does your agent decide which tool to call, and what stops it from looping forever?
- Walk me through what happens if the LLM tries to call a tool that doesn't exist / isn't allow-listed.
- How do you prevent a prompt-injection attack from a retrieved document triggering a write action?
- What's the blast radius if your MCP server's internal token leaks?
- Why plan→confirm→execute instead of just confirming with a "yes/no" text reply?
- RAG-specific: Why 384 dims here — what would change at 1536? How do you handle a query spanning two chunks? What does reranking solve that vector search alone doesn't?

### C. Node.js/runtime internals (if role is Node-heavy)
- Is Node single-threaded — then how does it stream to 1000 users at once?
- Difference between a generator, async generator, and a Node Readable stream — when do you use each?
- How do you avoid blocking the event loop when running an embedding model locally?
- How does `AbortController` propagate a client disconnect to an in-flight LLM call?

### D. DSA rounds — what they're actually testing
- Not "do you know the trick" — can you: clarify constraints, state a brute force, identify the pattern (two pointers/sliding window/DP/graph/etc.), code cleanly, test edge cases (empty input, single element, duplicates, negative numbers), and state time/space complexity unprompted.
- Practice explaining your approach *before* coding — most interviewers weight this as much as the working solution.
- Common trap: jumping to code before agreeing on the problem/constraints with the interviewer.

### E. LLD rounds — sample prompts + what's actually assessed
- Sample prompts: Design a Parking Lot / LRU Cache / Rate Limiter / Elevator System / Splitwise / Vending Machine.
- Assessed: can you model entities and relationships correctly, apply SOLID without being told to, handle concurrency where relevant (e.g., thread-safe rate limiter), and produce clean interfaces before jumping to implementation.
- Common trap: over-engineering with unnecessary design patterns instead of matching pattern choice to an actual requirement.

### F. HLD/backend deep-dive rounds
- Sample prompts: Design a URL shortener / rate limiter / notification system / chat app / your own `SCALING_HLD.md` scenario re-derived for a new hypothetical.
- Also expect pure backend questions: explain isolation levels and a concrete anomaly each prevents; explain N+1 queries and how an ORM causes them; explain indexing trade-offs (read speed vs write cost); when would you choose DynamoDB over RDS for a given access pattern.
- Common trap: designing before clarifying scale (QPS, data size, read/write ratio) — always do back-of-envelope estimation first.

### G. Security (any round, especially "AI + write access" or backend-with-PII contexts)
- Why should the retrieval/answer path and the action/tool path never share an LLM turn?
- How do you avoid leaking secrets into an LLM's context or logs?
- How do you prevent SQL injection at the ORM and raw-query layer both?
- What's your audit trail story when there's no per-user auth yet?

### H. Behavioral
- Tell me about a time you disagreed with a technical decision — what did you do?
- Tell me about a bug that took a long time to find — what was your process?
- Tell me about a time you had to learn an unfamiliar technology quickly under deadline pressure.

---

## 5. Combined roadmap (ordered by interview leverage, not calendar time)

Each phase ends with a self-check — if you can't answer it cleanly, you're not done with that phase.

### Phase 1 — DSA pattern gaps (highest volume of rounds gated on this)
Close the ⬜ gaps from §3A in this order: Trees → Graphs (BFS/DFS/topological/union-find) → Dynamic Programming → Heaps → Intervals → Bit manipulation → Tries. Keep solving inside your existing `_DSA_NeetCode` structure.
**Self-check:** Given an unseen medium-difficulty problem, correctly identify the pattern within 2 minutes and code a working solution within 20-25.

### Phase 2 — LLD fluency
Drill SOLID + the 3 pattern families (creational/structural/behavioral) until you can name the right pattern for a given problem unprompted. Solve 5-6 of the classic LLD problems end-to-end (Parking Lot, LRU Cache, Rate Limiter, Splitwise, Elevator).
**Self-check:** Given a new LLD prompt, produce a class diagram (even as text/mermaid) within 10 minutes before writing any code.

### Phase 3 — HLD fluency using your own material as the anchor
Re-derive `SCALING_HLD.md`'s tiers from scratch, unaided, for a different hypothetical (e.g., "design a notification system for 5M users"). Then drill back-of-envelope estimation explicitly (QPS, storage, bandwidth math) since it's a distinct skill from knowing the concepts.
**Self-check:** 45-minute mock, no notes, covering requirements clarification, high-level design, data model, and how it breaks at 10x.

### Phase 4 — Backend deep-dive gaps
Nail down: isolation levels + anomalies cold, N+1 query detection/fixes, indexing trade-offs, connection pooling. Cross-reference your own Hibernate/JPA notes rather than learning from scratch.
**Self-check:** Explain why a `SELECT *` in a loop (N+1) is slow, and name two ways to fix it (eager fetch/join, batch fetching) without notes.

### Phase 5 — LLM & Transformer fundamentals (biggest AI-track theory gap)
Cover: tokenization (BPE), embeddings/pooling (already known), attention (Q/K/V, multi-head), positional encoding, autoregressive decoding, sampling (temperature/top-p/top-k), context window & KV cache, training stages (pretrain→SFT→RLHF/DPO), quantization basics.
**Self-check:** Explain without notes why an LLM can "forget" something from 3 messages ago, why longer context costs more, and what temperature=0 mechanically changes.

### Phase 6 — Agentic system design vocabulary + close your own documented-but-unbuilt gaps
Cover ReAct vs Plan-and-Execute vs Reflexion, multi-agent patterns, memory types. Then actually implement what you already designed in `IMPROVEMENT_GUIDE.md`: sentence-aware chunking, cross-encoder reranking, embedding/query caching, HTTP-level rate limiting — so you can say "I built" not "I read about."
**Self-check:** Given "build an agent that triages support tickets," sketch loop/tools/memory/stop-conditions in under 5 minutes; show a before/after retrieval quality difference from reranking.

### Phase 7 — Evaluation/observability + framework fluency
Cover tracing, LLM-as-judge eval, golden-set regression testing, cost-per-request accounting. Survey LangChain/LangGraph, LlamaIndex, CrewAI/AutoGen, OpenAI Agents SDK — know what each abstracts over your hand-rolled loop.
**Self-check:** Explain how you'd know your agent got *worse* after a prompt change before a user complains; explain what LangGraph gives you that `agentLoop.js` doesn't.

---

## 6. External practice resources (verified references)

| Resource | Use for |
|---|---|
| [system-design-primer](https://github.com/donnemartin/system-design-primer) | Full HLD topic index + worked example solutions (Pastebin, Twitter feed, web crawler, etc.) + a study-guide calibrated by timeline |
| [awesome-low-level-design](https://github.com/ashishps1/awesome-low-level-design) | OOP fundamentals, SOLID, all design pattern families, UML, concurrency patterns, and a large LLD problem bank (easy/medium/hard) |
| NeetCode 150 / Blind 75 | Standard DSA pattern coverage — use to fill the ⬜ gaps identified in §3A |
| Your own `SCALING_HLD.md` | Best HLD practice material you have — more valuable than a generic problem because you already reasoned through every tier |
| Your own `IMPROVEMENT_GUIDE.md` | Turn every documented-but-unbuilt item into a real implementation exercise |

---

## 7. Practice cadence (no calendar, just discipline)

- Before any interview: rehearse both project narratives (§2) out loud, not just in your head.
- After learning each new concept: immediately re-explain it using a piece of *your own* code/notes as the example — theory sticks when tied to something you built.
- Run one system-design mock (Phase 3) and one LLD mock (Phase 2) per week of active prep, alternating AI-system and classic-system prompts.
- Timebox DSA to daily short sessions rather than long infrequent ones — pattern recall degrades fast without repetition.
- Keep a running doc of questions you fumbled — that list is your real study plan, more than this roadmap.

---

## 8. What "ready for any task" looks like (exit criteria)

You're ready when you can, unaided:
1. Solve an unseen medium DSA problem by correctly identifying the pattern, not by having memorized it.
2. Design a class model (LLD) for an unfamiliar domain and defend every SOLID/pattern choice.
3. Design a system (HLD) for an unfamiliar domain, do back-of-envelope estimation, and reason about what breaks at 10x/100x scale.
4. Explain isolation levels, N+1 queries, and indexing trade-offs cold, tied to your own Hibernate/JDBC experience.
5. Design an agent (tools, memory, loop, stop conditions) for an unfamiliar domain in one sitting.
6. Point to a concrete security mitigation for prompt injection, SQL injection, credential leakage, and runaway tool loops.
7. Explain every architectural decision in your own AI project and defend an alternative someone proposes.

---

_Next: pick a section (§3A-G) or a phase (§5, Phase 1-7) and we go deep, using your own code/docs/DSA folders as the worked example._
