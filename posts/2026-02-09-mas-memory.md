---
title: "Memory in LLM Agents: From Topology to Meta-Evolution"
date: 2026-02-09
author: "Sun Changsheng"
description: "Large Language Models are fundamentally stateless functions — they map an input $x$ to an output $y$, resetting with every call. To transform these static generators into autonomous agents capable of long-horizon reasoning and lifelong learning, we must equip them with **Memory**."
categories:
  - LLM Agents
  - Multi-Agent Systems
  - Memory Systems
tags:
  - memory
  - agents
  - multi-agent
  - cognitive-architecture
math: true
---

Large Language Models are fundamentally stateless functions — they map an input $x$ to an output $y$, resetting with every call. To transform these static generators into autonomous agents capable of long-horizon reasoning and lifelong learning, we must equip them with **Memory**.

While early approaches equated memory with simple context window stuffing or Retrieval-Augmented Generation (RAG), the field has moved toward far more sophisticated cognitive architectures. I've been reading a coherent series of papers (largely from Guibin Zhang and colleagues) that chart a trajectory from optimizing agent communication structures to generating latent cognitive states to meta-evolving the memory system itself. This post documents my learning process on this evolving landscape.

What makes this line of work interesting is that each piece addresses a limitation exposed by its predecessor, forming a chain of problem → solution → new problem:

| Stage | Work | The Question It Asks | Key Insight | What It Leaves Unsolved |
|:---|:---|:---|:---|:---|
| 1. Structure | **G-Designer** (2024) | How should agents communicate? | Topology is implicit memory — a VGAE can generate task-adaptive graphs, cutting 95% of token waste. | Only optimizes *connections*, not what each agent *does*. |
| 2. Structure | **MaAS** (2025) | What cognitive strategy should each agent use? | Model the workflow as an Agentic Supernet; sample simple paths for easy queries, complex ones for hard queries. | Agents are still *stateless* — everything is forgotten after a task. |
| 3. Explicit Memory | **G-Memory** (2025) | Where and how to store multi-agent experiences? | Flat retrieval fails for MAS. A three-tier graph (Interaction → Query → Insight) with bi-directional traversal captures both strategy and evidence. | Token-level storage is *slow* — every recall re-tokenizes text. |
| 4. Latent Memory | **MemGen** (2025) | Can we recall without reading text? | Inject *generated* latent vectors directly into attention — a "hippocampus" for the LLM. Faster, denser, no catastrophic forgetting. | Latent memories are *black boxes* — hard to debug. Also, what if stored experiences contain errors? |
| 5. Correction | **AgenTracer** (2025) | When a multi-agent workflow fails, *whose fault is it*? | Counterfactual replay pinpoints the decisive error step; store *corrected* trajectories, not raw failures. | Memory architecture still *hand-designed* per domain. |
| 6. Meta-Evolution | **MemEvolve** (2025) | Can the agent design its own memory system? | Decompose memory into Encode/Store/Retrieve/Manage modules; bi-level evolution searches over architectures. | Large memory graphs are computationally expensive. |
| 7. Infrastructure | **MoG** (ICLR 2025) | How to keep massive memory graphs tractable? | MoE-style node-level graph sparsification on the Grassmann manifold — principled forgetting at scale. | — |

[TOC]


# A Taxonomy for Agent Memory

Before diving into specific systems, it helps to establish a shared vocabulary. The survey *Memory in the Age of AI Agents* ([Hu et al. 2026](https://arxiv.org/abs/2512.13564)) points out that the field has long suffered from conceptual fragmentation — researchers mix terms like "long-term memory," "parametric memory," and "episodic memory" loosely, making it hard to compare approaches. Let me lay out a three-dimensional taxonomy that I find useful: *Forms*, *Functions*, and *Dynamics*.

## Forms: The Physical Substrate

"Form" determines the medium and information density of storage.

| Form | Definition | Pros | Cons | Examples |
|:---|:---|:---|:---|:---|
| **Token-Level** | Natural language text stored in external buffers or databases. | Human-readable, model-agnostic. | High token cost, read latency, context window limits. | G-Memory, MemGPT |
| **Parametric** | Information implicitly encoded in model weights via fine-tuning. | Zero read latency, deeply fused with reasoning. | Catastrophic forgetting, expensive updates, hard to edit. | MemoryLLM, Self-Param |
| **Latent** | Compressed into latent-space vectors or soft tokens. | Extremely high information density, machine-native, compute-efficient. | Human-unreadable, requires specialized decoder modules. | MemGen, MemoRAG |

The work we'll cover charts a clear migration from Token-Level (G-Memory) toward Latent (MemGen), driven by the pursuit of computational efficiency and reasoning coherence.

## Functions: What Memory Is For

Rather than the classical short-term / long-term split, a more useful functional taxonomy is:

- *Factual Memory*: Declarative knowledge about the world, users, and environment. The foundation of agent "honesty" — preventing hallucination.
- *Experiential Memory*: Procedural knowledge — "how to do things." Skills, insights, and workflows extracted from past successes and failures. The corrected trajectories from AgenTracer fall here.
- *Working Memory*: The active context for the current task. Dynamic filtering, folding, and compression to maintain reasoning coherence within a finite window.

## Dynamics: The Memory Lifecycle

Dynamics describe how memory flows through time:

- *Formation*: How key information is extracted from raw interaction streams (e.g., G-Memory distilling "insights" from dialogue).
- *Evolution*: How memories are consolidated, updated, or forgotten (the core innovation of MemEvolve).
- *Retrieval*: How the right information is accessed at the right time — from simple vector similarity to G-Memory's bi-directional graph traversal.



# The Structural Foundation: Topology as Implicit Memory

Before discussing *what* an agent remembers, we must consider *how* agents interact. In Multi-Agent Systems (MAS), the communication topology — who talks to whom and in what order — acts as a form of implicit working memory. If the structure is inefficient, information dissipates before it can be stored. Garbage in, garbage out — no downstream memory module can rescue a poorly structured information flow.

## G-Designer: Variational Graph Auto-Encoders for Communication

Standard multi-agent patterns (Chain, Star, Mesh) are static and suboptimal. A simple biology quiz doesn't need a fully connected graph; a complex math proof can't survive a single chain. **G-Designer** ([Zhang et al. 2024](https://arxiv.org/abs/2410.11782)) proposes treating agent topology generation as a graph generation problem, solved by a **Variational Graph Auto-Encoder (VGAE)**.

The setup: view agents as nodes in a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where node features $\mathbf{X}$ encode agent profiles (roles, tools, state) and the task query. A special *task node* broadcasts global information. The goal is to generate an adjacency matrix $\mathbf{A}$ representing the optimal communication structure.

The encoder $q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A}\_{\text{anchor}})$ maps agent features plus an initial anchor topology (like a simple chain) into a latent space $\mathbf{Z}$. The decoder $p(\mathbf{A} \mid \mathbf{Z})$ reconstructs a task-adaptive topology $\mathcal{G}\_{\text{com}}$. The optimization objective combines performance utility with sparsity regularization:

$$\mathcal{L} = \mathcal{L}\_{\text{utility}} + \beta_1 \|\mathcal{G}\| + \beta_2 \|\hat{\mathcal{G}}(\hat{Q}) - \mathcal{G}(Q)\|$$

Here $\|\mathcal{G}\|$ penalizes dense connections (reducing token cost), and the third term ensures robustness against adversarial perturbations on the query. The sparsity penalty is doing something quite interesting: it is essentially optimizing the system's "collective attention," ensuring that downstream memory modules only process high signal-to-noise information.

Results: G-Designer reduced token consumption by up to 95.33% on HumanEval compared to dense topologies like GPTSwarm, while maintaining 84.50% accuracy on MMLU. Dynamic structural pruning at the communication level turns out to be a prerequisite for effective memory.

## MaAS: Searching the Agentic Supernet

G-Designer optimizes connections, but it doesn't touch the operators inside each node — i.e., how each agent thinks. A natural follow-up is to also optimize the *cognitive strategy* at each node: should this agent use Chain-of-Thought? ReAct? Debate? **MaAS** (Multi-agent Architecture Search; [Zhang et al. 2025](https://arxiv.org/abs/2502.04180)) takes inspiration from Neural Architecture Search (NAS) to address exactly this.

The core concept is the **Agentic Supernet** — a continuous probability distribution over possible agentic operators (CoT, ReAct, Debate, Reflexion, etc.) arranged in a layered DAG. Each layer $\ell$ contains a set of candidate operators $\mathbb{O}$, and the probability of selecting operator $O$ is learned by a lightweight controller network parameterized by $\phi$:

$$\pi_\ell(O \mid \text{query}) = \text{Softmax}\big(\text{Controller}(v(\text{query}),\; v(O))\big)$$

A crucial innovation is making non-differentiable LLM calls differentiable. MaAS employs **textual gradients** — using LLMs to generate feedback on prompts and workflows — combined with Monte Carlo sampling to update the distribution parameters $\pi$. This allows adaptive sampling: simple architectures for easy queries (saving cost), complex ensembles for hard queries.

From a memory perspective, MaAS is a sophisticated form of *procedural memory*. It remembers "for this class of problem, invoke this reasoning paradigm," avoiding the cost of rebuilding workflows from scratch. Experiments show MaAS achieves higher accuracy on the MATH dataset at only 15% of the training cost and 25% of the inference cost compared to AFlow.

Together, G-Designer and MaAS solve the *structural* problem — agents now communicate efficiently and think with the right strategies. But both systems are still *stateless*. Once a task is done, everything is forgotten.



# Explicit Hierarchical Memory: G-Memory

Once agents are communicating effectively, they generate long, noisy interaction trajectories — often 10× longer than single-agent traces. Standard vector databases treat these as flat document collections, drowning in noise when trajectories stretch to tens of thousands of tokens. **G-Memory** ([Zhang et al. 2025](https://arxiv.org/abs/2506.07398)) argues that flat retrieval is fundamentally insufficient for MAS and proposes a hierarchical graph structure inspired by organizational memory theory.

## The Three-Tier Graph

G-Memory constructs three interconnected graph layers:

1. **Interaction Graph (Bottom Layer):** Stores fine-grained raw utterance chains — dialogue turns, tool call results, execution steps. This is the ground truth, the traceable evidence chain.

2. **Query Graph (Middle Layer):** Encodes task metadata and inter-task relationships. If "write a Python crawler" and "parse HTML" share structural similarity, their query nodes are linked. This layer captures *topological associations* between problems.

3. **Insight Graph (Top Layer):** Stores highly abstract, generalizable principles distilled from trajectories — for example, "When encountering a 403 error, check User-Agent first" or "For ALFWorld tasks, always clean objects before placing them." This is the crystallization of agent wisdom.

## Bi-Directional Traversal

The retrieval mechanism is where G-Memory truly differentiates itself from standard top-$k$ retrieval. When a new query $Q$ arrives, the system performs a **bi-directional traversal**:

- **Upward (inductive flow):** Locate similar queries in the Query Graph, then ascend to the Insight Graph. This retrieves the *Dao* — macro strategy and pitfall avoidance. "For math problems, always verify corner cases."
- **Downward (deductive flow):** Simultaneously descend to the Interaction Graph, extracting condensed core subgraphs. This retrieves the *Shu* — concrete code snippets, operation sequences, few-shot exemplars.

The combination mirrors how humans approach new problems: first recall general principles, then dig up specific past examples.

## Agentic Self-Update

G-Memory is not a static database. After every task, it performs an agentic refinement cycle:

- *Trajectory assimilation*: Absorb the new interaction record into the bottom layer.
- *Insight distillation*: Trigger a reflection process to extract new insights or update confidence scores on existing ones.
- *Association restructuring*: Dynamically adjust edge weights between query nodes based on task outcomes.

Experiments show a 20.89% improvement in success rates on embodied tasks (ALFWorld) and 10.12% on knowledge QA. The message: *structure* matters as much as *scale*.

That said, G-Memory operates entirely in the Token-Level form. Every retrieval requires re-tokenizing and re-attending to text — slow and expensive. This bottleneck motivates a paradigm shift.



# The Latent Turn: Generative Memory

A significant bottleneck in retrieval-based memory is the physical cost of text. Retrieving text means re-tokenizing it, feeding it through attention layers, consuming precious context window space — every single time. The human brain doesn't work this way. When you recall how to ride a bicycle, you don't internally "read" a paragraph about balance and pedaling. You *reconstruct a cognitive state*.

**MemGen** ([Zhang & Fu et al. 2025](https://arxiv.org/abs/2509.24704)) proposes exactly this shift: from *retrieving text* to *generating latent states*. I would consider this the most exciting development in the series — and the point where agent memory stops being "an external database" and starts becoming "a cognitive organ."

## MemGen: Trigger and Weaver

MemGen adds two lightweight modules alongside a frozen base LLM (the *Reasoner* $\pi_\theta$), deliberately leaving the core model untouched to prevent catastrophic forgetting. The design elegantly mirrors the hippocampal-cortical interaction in the brain — what cognitive scientists call **Complementary Learning Systems (CLS)** theory.

**Memory Trigger ($\mathcal{T}$):** A lightweight classifier (often implemented via LoRA) that monitors the Reasoner's hidden states $H_{t,\lt j}$ in real time. It acts as a metacognitive monitor: "Are you stuck? Do you need to recall something?"

$$p_j = \sigma\big(\mathcal{T}(H_{t,\lt j})\big)$$

If $p_j$ exceeds a learned threshold, reasoning pauses and the Trigger emits an `INVOKE` signal. The Trigger is trained via reinforcement learning, with rewards for invoking memory at genuinely critical decision points and penalties for wasteful triggers.

**Memory Weaver ($\mathcal{W}$):** Once triggered, the Weaver receives the current context state as a stimulus and *synthesizes* a sequence of **latent memory tokens** $\mathbf{M}_t$:

$$\mathbf{M}_t = \mathcal{W}(H_{t,\lt j})$$

These tokens are injected directly into the Reasoner's attention mechanism. Unlike RAG, which retrieves existing text chunks, the Weaver *generates* a memory representation tailored to the current context — potentially synthesizing information from multiple past experiences stored implicitly in its weights and any retrieved external cues. One latent token can encode the semantic information of hundreds of text tokens.

## Emergent Cognitive Specialization

The most striking finding in MemGen is the *spontaneous emergence* of functionally specialized memory types without explicit supervision:

- *Planning Memory:* Certain latent tokens activate specifically during task decomposition phases.
- *Procedural Memory:* Activates during tool calls and code generation — analogous to muscle memory.
- *Working Memory:* Activates during long-document processing to maintain cross-paragraph coherence.

This emergent specialization is a cool result. It suggests that with the right architectural inductive bias (separate Reasoner and Weaver, triggered injection), the system naturally develops cognitive divisions of labor.

Experiments show MemGen outperforms dedicated parametric memory methods like ExpeL and AWM by up to 38.22% on cross-domain tasks, while fully solving the catastrophic forgetting problem — the Reasoner stays frozen; only the Weaver adapts to new domains.

The tradeoff is interpretability. A human cannot inspect a latent vector and understand what the agent "remembered." The memory is a black box. This is a real limitation for debugging and safety auditing, but one that the community will likely address through probing and visualization techniques.

MemGen solves the *efficiency* problem. But it introduces a new concern: what if the experiences being stored — whether explicitly in G-Memory or implicitly in the Weaver's weights — contain errors? A memory system that faithfully records hallucinations will poison its own well.



# Correction and Diagnostics: AgenTracer

*Superstitious learning* — storing spurious correlations as "experience" — is a real risk when agents reflect on noisy trajectories. **AgenTracer** ([Zhang et al. 2025](https://arxiv.org/abs/2509.03312)) addresses the blame assignment problem: when a complex multi-agent workflow fails after dozens of steps, *which step caused it?*

The method uses **Counterfactual Replay**. For a failed trajectory $\tau$, AgenTracer systematically intervenes at each candidate error step $t$, replacing the agent's action $a_t$ with an Oracle (correct) action $a'_t$, and replays:

$$C(\tau) = \{(i, t) \mid \Omega(\tau) = 0 \;\land\; \Omega(\mathcal{R}(\tau, t, a'_t)) = 1 \}$$

where $\Omega(\cdot)$ returns 1 for success and 0 for failure, and $\mathcal{R}(\tau, t, a'_t)$ is the trajectory with the intervention at step $t$. If the system succeeds after intervention, step $t$ is the **root cause**. If the failure persists, the error lies downstream.

AgenTracer also works in reverse: injecting noise into *successful* trajectories to verify that identified critical steps are truly decisive, not coincidental.

The output is a curated dataset of failure-correction pairs (*TracerTraj*). By storing *corrected trajectories* rather than raw error trajectories into memory, AgenTracer implements grounded self-evolution — memory that actively cleans itself. Integrating AgenTracer feedback into off-the-shelf frameworks like MetaGPT yielded 4.8%–14.2% performance improvements. This marks a transition from passive recording to active curation of experience.



# Meta-Evolution: MemEvolve

Everything discussed so far — G-Memory's three-tier graph, MemGen's latent injection, AgenTracer's counterfactual cleaning — uses a *fixed* architecture chosen by the developer. But different task domains have radically different memory needs. Creative writing benefits from associative, loosely connected memories; code debugging needs precise, keyword-matched procedure recall.

**MemEvolve** ([Hu et al. 2025](https://arxiv.org/abs/2512.18746)) treats the memory architecture itself as a variable to be optimized — what the authors call **meta-evolution of the memory architecture**.

## The EvolveLab Design Space

MemEvolve first decomposes any memory system into four standardized, swappable modules:

- **Encode:** How to transform raw experience into storage format (Summarize? Embed? Extract code snippets? Distill into rules?).
- **Store:** Where to put it (JSON file? Vector database? Knowledge graph? Key-value store?).
- **Retrieve:** How to access it (Keyword matching? Semantic similarity? Hybrid? Graph traversal?).
- **Manage:** Pruning and forgetting policies (FIFO? Importance-based? Clustering? Time-decay?).

Each module has multiple candidate implementations, forming a combinatorial search space.

## Bi-Level Optimization

MemEvolve runs a bi-level evolutionary loop:

- **Inner loop (experience evolution):** The agent executes tasks under a *fixed* memory architecture, accumulating experience $M_t$.
- **Outer loop (architecture evolution):** Based on inner-loop performance (success rate, token cost, latency), the system applies Pareto-optimal mutation and selection *on the memory architecture itself*.

For example, a web browsing task might evolve toward "store API call sequences + exact-match retrieval," while a math reasoning task might evolve toward "store error reflections + semantic retrieval." The system transitions from a *skillful learner* (fixed architecture, improving content) to an *adaptive learner* (evolving both architecture and content).

Experiments show that architectures evolved on TaskCraft transfer directly to unseen benchmarks like GAIA and xBench with significant performance gains, and SmolAgent performance improved by 17.06%. The practical implication: we may soon stop hand-tuning memory hyperparameters entirely.


# Efficiency at Scale: Mixture of Graphs

As graph-based memory systems like G-Memory accumulate interaction histories, the graph size explodes. Running operations on massive graphs becomes a computational bottleneck.

**Mixture-of-Graphs (MoG)** ([Zhang et al., ICLR 2025](https://arxiv.org/abs/2405.14260)) introduces a principled sparsification technique. While originally designed for GNN acceleration, its application to maintaining massive memory graphs is a natural extension.

The core insight: different nodes live in different local contexts, so a single global pruning rule (e.g., "remove low-weight edges") is suboptimal. MoG trains multiple **Sparsifier Experts**, each applying a different pruning criterion (gradient magnitude, Jaccard similarity, effective resistance). For each node, a router selects the most appropriate expert, producing a locally optimized sparse subgraph:

$$\mathcal{G}^{(i)}_{\text{sparse}} = \text{Ensemble}\Big(\left\{ \kappa_m(\mathcal{G}^{(i)}) \right\}_{m=1}^k\Big)$$

The expert outputs are aggregated on the **Grassmann manifold** to preserve spectral properties of the graph — a technically elegant choice that ensures the sparsified graph retains the essential information-propagation characteristics of the original.

MoG achieves 50% graph sparsity while maintaining or even improving model performance, with 1.47–2.62× inference speedup. In the memory context, this provides a principled *forgetting mechanism*: intelligently pruning redundant, low-value memory associations while preserving critical reasoning pathways.



# What's for Future

The trajectory of this research highlights a maturation of the "Agent" concept. We are moving away from treating LLMs as standalone oracles toward treating them as CPUs within a larger cognitive architecture. Let me highlight specific directions I find most promising.

**From retrieval to generation as the default.** MemGen opens a door that I believe will remain wide open. Future agents will not "consult" past diary entries; they will *reconstruct cognitive states* by activating learned neural circuits. This reconstructive memory is more biologically plausible and more computationally efficient. The key research challenge is interpretability — we need probing techniques that let us understand what latent memory tokens encode.

**Automated memory management as standard practice.** Hand-tuning chunk sizes, top-$k$ values, and retrieval strategies should become obsolete. MemEvolve's approach — treating memory architecture as a searchable variable — will likely become the norm. I would not be surprised to see dedicated *Memory Architect* meta-agents whose sole job is monitoring and optimizing the working agents' memory structures.

**Memory consolidation via online RL.** The boundary between parametric and non-parametric memory will blur. Short-term database experiences will be periodically "compiled" into lightweight model adapters via online reinforcement learning — a mechanism analogous to human sleep consolidation, transforming episodic memory into semantic memory. MemGen's Weaver already hints at this direction.

**Shared cognitive substrates in MAS.** Currently each agent maintains isolated memory. The natural next step is distributed shared memory built on G-Memory's hierarchical graph, evolving into a kind of "collective consciousness" for agent clusters — with privacy and permission filters (analogous to G-Designer's topological constraints) to manage access.

I especially want to call out the convergence that seems inevitable: a system that dynamically designs its own topology (MaAS), uses AgenTracer to filter experiences, stores them in a hierarchical G-Memory, recalls them as latent states via MemGen, and continuously evolves its memory architecture through MemEvolve. The pieces are all here; the integration is the next frontier.



## Summary Comparison

| Dimension | G-Memory | MemGen | MemEvolve |
|:---|:---|:---|:---|
| **Form** | Explicit text (Graph) | Latent vectors | Dynamic / evolvable |
| **Core Mechanism** | Three-tier graph + bi-directional traversal | Trigger + Weaver injection | Bi-level evolutionary loop |
| **Interpretability** | High (human-readable) | Low (black box) | Medium (architecture is inspectable) |
| **Inference Speed** | Slow (re-tokenize context) | Fast (direct attention injection) | Depends on evolved architecture |
| **Key Problem Solved** | Long-horizon dependencies, structured information | Efficiency, catastrophic forgetting | Manual architecture design cost |

<!-- --- -->

<!-- *Cited as:*

```bibtex
@article{weng2025memory,
  title   = "Memory in LLM Agents: From Topology to Meta-Evolution",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io",
  year    = "2025",
  month   = "Jun",
  url     = "https://lilianweng.github.io/posts/2025-06-15-agent-memory/"
}
``` -->

## References

[1] Guibin Zhang et al. ["Graph Sparsification via Mixture of Graphs."](https://arxiv.org/abs/2405.14260) **ICLR 2025 Spotlight**.

[2] Guibin Zhang et al. ["G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks."](https://arxiv.org/abs/2410.11782) **ICML 2025 Spotlight**.

[3] Guibin Zhang et al. ["Multi-agent Architecture Search via Agentic Supernet."](https://arxiv.org/abs/2502.04180) **ICML 2025 Oral**.

[4] Guibin Zhang et al. ["G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems."](https://arxiv.org/abs/2506.07398) **NeurIPS 2025 Spotlight**.

[5] Guibin Zhang et al. ["AgenTracer: Who is Inducing Failure in the LLM Agentic Systems?"](https://arxiv.org/abs/2509.03312) **arXiv:2509.03312. Preprints**.

[6] Guibin Zhang, Zhihao Fu et al. ["MemGen: Weaving Generative Latent Memory for Self-Evolving Agents."](https://arxiv.org/abs/2509.24704) **arXiv:2509.24704. Preprints**.

[7] Zijian Hu et al. ["Memory in the Age of AI Agents: A Survey."](https://arxiv.org/abs/2512.13564) **arXiv:2512.13564. Preprints**.

[8] Zijian Hu et al. ["MemEvolve: Meta-Evolution of Agent Memory Systems."](https://arxiv.org/abs/2512.18746) **arXiv:2512.18746. Preprints**.