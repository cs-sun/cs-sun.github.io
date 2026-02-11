---
title: "When Graphs Meet Agents: Orchestration, Topology, and the Uncharted Territory of Safety"
date: 2026-02-09
author: "Sun Changsheng"
description: "A deep dive into the emerging intersection of graph structures and LLM agentic systems, exploring orchestration frameworks, topology-aware design, and the largely unexplored frontier of topology-based security."
categories:
  - LLM Agents
  - Graph Neural Networks
  - AI Security
  - Multi-Agent Systems
tags:
  - survey
  - graphs
  - agent-orchestration
  - GNN
  - security
math: true
---

<!-- [TOC] -->

# Why Graphs?

If you squint at any sufficiently complex LLM agent system — one that plans, decomposes tasks, dispatches tool calls, and synthesizes results — you'll notice it's a graph. Not metaphorically. Literally. There are nodes (tasks, agents, LLM calls) and edges (dependencies, information flow, communication channels). The question is whether we should treat this graph structure as an incidental implementation detail or as a first-class object worthy of optimization, analysis, and — crucially — adversarial scrutiny.

Over the past two years, a growing body of work has chosen the latter. Papers like **LLMCompiler** ([Kim et al., 2024](https://arxiv.org/abs/2312.04511)), **GPTSwarm** ([Zhuge et al., 2024](https://arxiv.org/abs/2402.16823)), and **LocAgent** ([2025](https://arxiv.org/abs/2503.09089)) explicitly construct, manipulate, and optimize graph structures to orchestrate agentic behavior. Meanwhile, a parallel (and largely disconnected) literature has been studying how to attack and defend LLM agents — prompt injections, jailbreak propagation, tool misuse. But here's what struck me: almost nobody is asking what happens when you attack the *graph itself*. Not the prompts flowing through edges, but the topology, the wiring, the structure of who-talks-to-whom and which-task-depends-on-which.

This post is my attempt to map the landscape at this intersection: how graphs are used in agentic AI, what we know about how topology affects both performance and safety, and where the most promising (and most vacant) research real estate lies.

# The Rise of Graph-Based Orchestration

## DAGs as the Backbone of Parallel Execution

The simplest and most intuitive use of graphs in agentic systems is to model task dependencies as directed acyclic graphs (DAGs). If task B depends on the output of task A, but task C is independent of both, a DAG captures this cleanly and enables parallel execution of A and C.

**LLMCompiler** ([Kim et al., 2024](https://arxiv.org/abs/2312.04511)) was among the first to formalize this idea for LLM function calling. The architecture is elegant: a planner LLM generates a task dependency DAG, a dispatcher executes tasks in topological order (parallelizing independent branches), and a joiner synthesizes results. The payoff is substantial — 3.7× latency speedup and up to 1.35× accuracy improvement over ReAct-style sequential execution on benchmarks. The key insight is that many real-world queries contain inherent parallelism that sequential chain-of-thought reasoning wastes.

**StateFlow** ([Wu et al., 2024](https://arxiv.org/abs/2403.11322)) takes a complementary angle, modeling agentic workflows as finite state machines — a special case of directed graphs where nodes represent execution states and edges represent transitions. This formulation is less about parallelism and more about *control*: by explicitly defining legal state transitions, StateFlow constrains the agent's behavior in ways that free-form prompting cannot. The results bear this out: 13–28% higher success rates than ReAct at 3–5× lower cost. I find this particularly interesting because it highlights a tension that runs through the entire field: expressiveness vs. controllability. The more structure you impose on the execution graph, the less you rely on the LLM to "figure it out" — and the more predictable (and verifiable) the system becomes.

For scenarios where task dependencies evolve during execution, **DynTaskMAS** (2025) introduced continuously-updated dynamic task DAGs for asynchronous multi-agent execution. And on the automated design front, **AFlow** ([Zhang et al., 2025a](https://arxiv.org/abs/2410.10762)) uses Monte Carlo Tree Search over two-level graph abstractions to automatically generate workflow graphs, removing humans from the loop of pipeline design entirely.

## Agents as Computational Graphs

A more radical perspective — and one I think has deeper implications — is to treat the entire agent system as a *differentiable computational graph*. **GPTSwarm** ([Zhuge et al., 2024](https://arxiv.org/abs/2402.16823)), presented as an oral at ICML 2024, unifies diverse agent paradigms under this framework. Chain-of-Thought, Tree-of-Thought, Reflexion — these aren't different algorithms so much as different *graph topologies* over LLM-invoked functions. Nodes are operations (LLM calls, tool invocations, aggregation functions), and edges carry information between them. The punchline: you can optimize the edge weights via reinforcement learning, and the system discovers high-performance configurations that resemble — or improve upon — hand-designed agent architectures.

Why does this matter? Because it shifts agent design from a craft (prompt engineering, hand-coded pipelines) to a *search problem over graph space*. And once you're searching over graph space, all the machinery of graph optimization, graph neural networks, and — yes — graph adversarial robustness becomes directly applicable.

## Domain-Specific Graph Agents

Among the papers I've been reading, **LocAgent** ([2025](https://arxiv.org/abs/2503.09089)) stands out for how cleanly it demonstrates the value of explicit graph structure. The task is code localization — given a bug report, find the relevant files and functions in a large codebase. LocAgent constructs a directed heterogeneous graph over the codebase with four edge types: *contain*, *import*, *invoke*, and *inherit*. Agents are then equipped with graph traversal tools for multi-hop dependency reasoning. The result: 92.7% file-level localization accuracy at 86% cost reduction on SWE-bench-Lite. The graph here isn't just scaffolding; it encodes domain knowledge about code structure that would be expensive and unreliable to recover from raw text alone.

**OpenHands** ([Wang et al., 2025a](https://arxiv.org/abs/2407.16741)), the open-source agentic coding platform (formerly OpenDevin), takes a different architectural path. Rather than explicit graph scheduling, it uses an event stream architecture with hierarchical agent delegation — tree-like in structure, but without formal graph-based optimization. This is representative of many production systems: they *implicitly* create graph structures through delegation and composition, even if the graph is never explicitly constructed or analyzed.

## What Production Deployments Taught Us

I want to pause on a piece of practical wisdom that I think is under-appreciated in the academic literature. **Atlassian's engineering team**, in deploying their Rovo Chat multi-agent system ([Atlassian, 2025](https://www.atlassian.com/blog/atlassian-engineering/how-rovo-embraces-multi-agent-orchestration)), found that single-shot DAG planning from user queries is *brittle*. Their initial approach — have an LLM look at the user query and output a complete execution DAG — worked well on clean benchmarks but fell apart on the messy, ambiguous queries real users produce. They reverted to hybrid orchestrators that combine graph-based scheduling with fallback heuristics.

This suggests something important: DAG orchestration may be better suited for *post-training optimization* (where you have time to search over graph structures, as in AFlow or GPTSwarm) than for *runtime planning* (where a single LLM call must produce a correct DAG from an ambiguous query). It's a lesson about the difference between the graph as a *design artifact* vs. the graph as a *runtime prediction*.

# Does the Shape of the Graph Matter?

This is perhaps the most fundamental question in graph-based agentic AI, and the answer turns out to be both "yes, enormously" and "it's complicated."

## No Topology Rules Them All

**MacNet** ([Qian et al., 2025](https://openreview.net/forum?id=K3n5jPkrU6)), accepted at ICLR 2025, conducted the first systematic comparison of communication topologies for multi-agent LLM collaboration. They tested an impressive array: chain, tree, star, binary tree, mesh, layered, random, and complete graph configurations, across heterogeneous tasks including software development, logical reasoning, and math.

The headline finding: **no single topology is universally optimal**. Chains excel for software development (where sequential refinement matters), while mesh topologies dominate logical selection tasks (where diverse perspectives help). This isn't entirely surprising — it mirrors findings in distributed systems and organizational theory — but the paper quantifies it rigorously. MacNet also discovered a *collaborative scaling law* following logistic growth: collaborative emergence (where multi-agent systems start outperforming single agents) occurs at smaller scales than neural emergence (where bigger models start showing qualitative improvements). This is encouraging — it suggests you don't need hundreds of agents to see benefits from topology design.

## Learning the Right Topology with GNNs

If no fixed topology works everywhere, the natural next step is to *learn* the topology. **G-Designer** ([Zhang et al., 2025b](https://arxiv.org/abs/2410.11782)), accepted at ICML 2025, does exactly this using a variational graph auto-encoder (VGAE). Given a task description, G-Designer predicts the optimal agent communication graph — which agents should talk to whom, and with what connectivity pattern. It consistently outperforms static hand-crafted topologies, achieving up to 95% token reduction (!) while maintaining accuracy. The token reduction is particularly noteworthy: not only does the learned topology perform better, but it communicates *far less*, because it routes information only where it's needed.

More recently, **Guided Topology Diffusion (GTD)** ([2025](https://arxiv.org/abs/2510.07799)) applied conditional discrete graph diffusion with a Graph Transformer denoiser to generate communication graphs, achieving Pareto-optimal accuracy-cost tradeoffs. **Assemble Your Crew** ([2025](https://arxiv.org/abs/2507.18224)) further advances this line with autoregressive graph generation for topology design. The trend is clear: topology design is moving from hand-crafted to learned, and GNNs are the method of choice.

I want to emphasize something that often gets lost in the method details: these papers are demonstrating that *the structure of agent interaction matters as much as the content*. You can have the best LLM in the world, but if you wire it into a suboptimal communication topology, you'll get suboptimal results. This is a fundamentally graph-theoretic insight.

## GNNs Can Predict Whether Your Workflow Will Succeed

A pivotal bridging result — and one that I think opens the door to a whole class of new research — comes from **FLORA-Bench** ([Zhang et al., 2025c](https://arxiv.org/abs/2503.11301)). This paper asks a deceptively simple question: if you represent an agentic workflow as a computational DAG, can a GNN predict whether it will succeed?

The answer is yes, with >0.8 accuracy across different LLM backends (GPT-4o-mini, DeepSeek V3, Qwen 7B). Standard GNN architectures — GCN, GAT, GCNII, Graph Transformer — all work, and they significantly outperform LLMs that are given serialized text descriptions of the same workflows. **GLOW** ([2025](https://arxiv.org/abs/2512.15751)) extends this further by integrating GNN structural inductive bias with LLM semantic expressiveness, reaching 99.1% workflow parsing accuracy.

Why do I consider this a pivotal result? Because it establishes that **GNNs are already processing agentic workflow graphs as their native input**. This means the entire toolkit of GNN robustness, GNN explainability, and GNN adversarial attacks becomes *directly applicable* — not through some forced analogy, but through the natural problem formulation. If a GNN predicts your workflow will succeed, you can ask: *how robust is that prediction to small perturbations of the graph?* And nobody, as far as I can tell, has asked that question yet.

# The Security Landscape: Mostly Content, Rarely Structure

The LLM agent security literature has grown explosively, but if you read it carefully, you'll notice a pattern: almost all attacks target *content* (prompts, messages, tool inputs), and almost all defenses operate on *content* (output filtering, input sanitization, safety classifiers). The graph structure — the wiring of the system — is treated as immutable background.

## How Agents Get Attacked

The attack surface of multi-agent LLM systems is larger than most people realize. **Agent Smith** ([Gu et al., 2024](https://arxiv.org/abs/2402.08567)), presented at ICML 2024, demonstrated something genuinely alarming: a single adversarial image injected into one agent can trigger *exponentially fast* jailbreak propagation across an entire multi-agent network. The mechanism is viral — compromised agents produce adversarial outputs that compromise their neighbors, who in turn compromise *their* neighbors. The exponential growth rate means that by the time you detect the attack, most of the network may already be compromised.

**Prompt Infection** (2024) studied self-replicating prompt injections that spread between communicating agents. The **TAMAS** benchmark ([2025](https://arxiv.org/abs/2511.05269)) systematically evaluates six attack types — including Byzantine agents (which behave arbitrarily) and colluding agents (which coordinate their attacks) — across centralized, decentralized, and swarm configurations. **Agent Security Bench (ASB)** ([2025](https://openreview.net/forum?id=V4y0CpX4hK)), at ICLR 2025, provides the most comprehensive evaluation to date: 27 attack/defense methods across ~90K test cases, with maximum attack success rates reaching 84.30%.

And these aren't just theoretical concerns. **OpenHands**, one of the most popular open-source agentic platforms, suffered two critical exploits in 2025: a zero-click data exfiltration attack where an adversarial image could steal `GITHUB_TOKEN` values via URL parameters ([Embrace The Red, 2025a](https://embracethered.com/blog/posts/2025/openhands-the-lethal-trifecta-strikes-again/)), and a remote code execution exploit where untrusted web data could hijack the agent to connect to attacker-controlled servers ([Embrace The Red, 2025b](https://embracethered.com/blog/posts/2025/openhands-remote-code-execution-zombai/)). Browser agent testing found OpenHands executed harmful behaviors in 67 out of 100 test cases without guardrails ([Invariant Labs, 2025](https://invariantlabs.ai/blog/enhancing-browser-agent-safety)).

## Defenses: From Guardrails to Formal Guarantees

Defense approaches span a wide spectrum of rigor. At the formal end, **CaMeL** ([Debenedetti et al., 2025](https://arxiv.org/abs/2503.18813)) from Google DeepMind offers *provable* security against prompt injection. The key idea is elegant: extract control-flow and data-flow graphs from the *trusted* query (what the user actually asked for), then enforce at runtime that untrusted data (web pages, tool outputs, etc.) can never influence program flow. If the data-flow graph shows that an untrusted string would be used in a control decision, CaMeL blocks it. This achieves provable security on the AgentDojo benchmark.

A companion paper on **Design Patterns for Securing LLM Agents** ([Asan et al., 2025](https://arxiv.org/abs/2506.08837)) from IBM/ETH/Google/Microsoft proposes six architectural patterns — Plan-then-Execute, LLM Map-Reduce, Dual LLM — that constrain the workflow graph to resist prompt injection *by construction*. I find this approach compelling because it shifts security from an afterthought (add a filter) to a *structural property* of the workflow graph itself.

Other notable defenses include **TrustAgent** ([Hua et al., 2024](https://arxiv.org/abs/2402.01586)), which introduced an "Agent Constitution" with pre-planning, in-planning, and post-planning safety checks; **VeriGuard** ([2025](https://arxiv.org/abs/2510.05156)), which applied Hoare-triple-based formal verification to individual agent actions; and **AgentSpec** (2026), which provides a domain-specific language for runtime safety rule enforcement.

On the LLM capability side, **R-Tuning** ([Zhang et al., 2024](https://arxiv.org/abs/2311.09677)), which won the Outstanding Paper award at NAACL 2024, trains LLMs to recognize their knowledge boundaries and say "I don't know." While not graph-related, this capability is directly relevant to agentic safety — an agent that knows when it's uncertain can avoid taking catastrophic actions on shaky reasoning.

## When Reasoning Itself Becomes the Vulnerability

One paper that sits at an interesting angle to the graph-based perspective is **"The Danger of Overthinking"** ([2025](https://arxiv.org/abs/2502.08235)), which identifies the *reasoning-action dilemma* in agentic tasks. Agents using extended chain-of-thought reasoning can fall into failure modes where excessive reasoning leads to "Rogue Actions" — the agent violates sequential execution constraints, essentially acting before it has finished thinking, or thinking itself into an inconsistent state.

This is deeply relevant to graph-based scheduling. A workflow DAG imposes an execution order — task B shouldn't start before task A finishes. But if an agent within a node can unilaterally decide to skip ahead or re-order operations based on its internal reasoning, the DAG's guarantees are violated from within. It's a reminder that graph-level safety requires not just structural integrity of the DAG, but also behavioral compliance of the agents *within* each node.

# Topology-Aware Security: A Frontier with Three Papers

Here's the punchline of this survey, and the reason I wrote this post: **the intersection of graph topology and security in multi-agent LLM systems is almost completely unexplored.** I found exactly three papers that directly study how graph structure affects security — and they all come from the same (or closely related) research group.

## Detecting Attacks on the Graph

**G-Safeguard** ([Wang et al., 2025b](https://arxiv.org/abs/2502.11127)), published at ACL 2025, is the single most relevant paper to this intersection. It introduces the first security mechanism that operates *on the graph topology* of multi-agent systems. The approach has two phases: (1) train an edge-featured GNN to detect anomalous agents by analyzing the multi-agent utterance graph — the graph whose nodes are agents and whose edges carry conversation messages; and (2) perform *topological interventions* (essentially, graph pruning) to isolate compromised agents and remediate attacks.

The results are promising: G-Safeguard recovers over 40% performance under prompt injection attacks across tree, chain, and graph configurations, with generalizability across different LLM backbones. What I find most interesting is the conceptual framing — the paper demonstrates that adversarial influence leaves *structural signatures* in the communication graph that a GNN can detect, even when the textual content of messages looks benign to a naive classifier.

## Which Topologies Are Safer?

**NetSafe** ([Yu et al., 2025](https://arxiv.org/abs/2410.15686)), also at ACL 2025, directly studies the relationship between topological properties and safety. The findings are intuitive in retrospect but had not been previously quantified: highly connected networks are more susceptible to adversarial spread (because there are more paths for malicious influence to propagate); star topology performance drops by 29.7% under attack (because the central node is a single point of failure); and networks with greater average distance from the attacker exhibit enhanced safety (because adversarial influence attenuates over multiple hops).

NetSafe also identifies two interesting phenomena: "Agent Hallucination" (agents producing unreliable outputs under adversarial influence from topological neighbors) and "Aggregation Safety" (the degree to which different aggregation strategies — majority vote, weighted average, etc. — amplify or dampen adversarial signals).

**MAMA** ([2024](https://arxiv.org/abs/2512.04668)) complements this by studying PII (Personally Identifiable Information) leakage across six canonical topologies: complete, ring, chain, tree, star, and star-ring. The central result aligns with NetSafe: denser connectivity and shorter attacker-target distances increase leakage. Fully connected and star-ring topologies are most vulnerable; chains and trees offer the strongest privacy protection. The topology ordering is preserved across different LLM backbones, suggesting this is a structural property rather than a model-specific artifact.

These three papers together establish a clear principle: **topology is a security-relevant design choice, not just a performance-relevant one.** Choosing a fully connected communication graph for your multi-agent system isn't just wasteful in tokens — it's actively *dangerous* because it maximizes the attack surface.

## Risk-Aware Composition: Almost There

One paper that partially enters the risk-optimization territory deserves special mention. **"Risk-Sensitive Agent Compositions"** ([2025](https://arxiv.org/abs/2506.04632)) formalizes agentic workflows as DAGs called "agent graphs" where edges represent agents and paths represent feasible compositions. It then minimizes Value-at-Risk (VaR) of loss distributions encoding safety, fairness, and privacy violations.

This is conceptually close to what I think the field needs, but it stops short in several ways: it uses VaR rather than CVaR (Conditional Value-at-Risk, which better captures tail risk), it doesn't employ distributionally robust optimization (DRO), and it doesn't leverage GNN-based workflow prediction. There's a natural extension here that combines the GNN workflow predictors from FLORA-Bench with CVaR-based risk optimization — but nobody has done it yet.

# What's Missing: Research Gaps I Find Exciting

After reading through this literature, I'm struck by how wide the gaps are. Here are the directions that seem most promising to me, roughly ordered by how excited I am about them.

**Certified robustness of workflow graphs.** We have certified robustness methods for GNNs — randomized smoothing on graphs, spectral certificates, certified perturbation bounds. We have GNNs that predict agentic workflow performance (FLORA-Bench, GLOW). Nobody has connected the two. The question "if I add or remove $k$ edges from this workflow DAG, is the GNN's performance prediction still valid?" is both well-defined and practically important, and it has never been asked.

**Adversarial structural attacks on workflow DAGs.** The graph adversarial robustness community has a mature toolkit of structural attacks — Metattack, Nettack, PGD on graph structure. The question "how many edge perturbations does it take to make a workflow DAG fail?" is the agentic-AI analog of these attacks. This would reveal which workflow structures are inherently fragile and which are robust — information that's critical for deploying agentic systems in high-stakes settings.

**GNN-explainable agent orchestration.** If GNNs can predict whether a workflow will succeed, the natural follow-up is: *which substructures drive that prediction?* Applying GNNExplainer, SubgraphX, or concept-based graph explanations to workflow predictors would tell us which parts of a workflow DAG are critical for success and which are redundant. This has both debugging applications (why did this workflow fail?) and security applications (which substructures, if perturbed, would cause failure?).

**CVaR/DRO-optimized workflow graphs.** The risk-sensitive agent compositions paper uses VaR, but CVaR (expected tail loss) is a more appropriate risk measure for safety-critical systems because it accounts for the *severity* of worst-case outcomes, not just their probability. Combining CVaR optimization with GNN-predicted workflow performance under distributional uncertainty (Wasserstein ambiguity sets, for instance) would yield workflows that are provably robust under worst-case distributional shifts.

**Directed vs. undirected graph impact on safety.** Here's a subtle but important gap: all the topology-safety papers (G-Safeguard, NetSafe, MAMA) study *communication graphs*, which are typically undirected or bidirectional (agent A can message agent B, and vice versa). But workflow DAGs — the task dependency structures — are *directed* and *acyclic*. The directional constraint fundamentally changes the attack surface: in a DAG, adversarial influence can only propagate forward (downstream), not backward. This asymmetry hasn't been studied, and it likely has significant implications for defense design.

**Formal verification of workflow DAGs.** VeriPlan ([2025](https://dl.acm.org/doi/10.1145/3706598.3714113)) verifies sequential agent plans, but nobody has applied model checking or formal verification to graph-structured workflows. Properties like deadlock freedom (no circular waits in the task graph), guaranteed task completion (all paths lead to a terminal node), and Byzantine fault tolerance (the workflow produces correct output even if $k$ agents are compromised) are well-defined graph properties amenable to tools like TLA+ or SPIN. The gap between formal methods and agentic AI remains wide.


*Conclusion.* The landscape I've described is one of rapid, somewhat fragmented progress. Graph-based orchestration has matured quickly — from concept to ICML/ICLR papers to production deployment in under two years. Agent security has grown explosively but remains fixated on content-level attacks. The critical missing piece is the bridge: applying the mature toolkit of graph adversarial robustness, GNN explainability, and structural risk optimization to the workflow graphs that increasingly define how agentic systems operate. The three papers at this intersection (G-Safeguard, NetSafe, MAMA) have opened the door, but the room beyond is largely empty.

<!-- For researchers with expertise in GNN robustness and graph-based optimization, this is — I think — an unusually attractive moment to enter the field. -->

# References

[1] Kim, S., Moon, S., Tabrizi, R., Lee, N., Mahoney, M.W., Keutzer, K., & Gholami, A. (2024). [An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511). *ICML 2024*.

[2] Wu, J., Zhang, Y., Li, L., & Wang, C. (2024). [StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows](https://arxiv.org/abs/2403.11322).

[3] DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous Multi-Agent Systems. *ICAPS 2025*.

[4] Zhang, J., et al. (2025a). [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762). *ICLR 2025*.

[5] Zhuge, M., et al. (2024). [GPTSwarm: Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823). *ICML 2024 (Oral)*.

[6] [LocAgent: Graph-Guided LLM Agents for Code Localization](https://arxiv.org/abs/2503.09089). *ACL 2025*.

[7] [LangGraph](https://github.com/langchain-ai/langgraph). LangChain.

[8] Atlassian Engineering. (2025). [How Rovo Chat Embraces Multi-Agent Orchestration](https://www.atlassian.com/blog/atlassian-engineering/how-rovo-embraces-multi-agent-orchestration).

[9] Qian, C., et al. (2025). [Scaling Large Language Model-based Multi-Agent Collaboration (MacNet)](https://openreview.net/forum?id=K3n5jPkrU6). *ICLR 2025*.

[10] Zhang, G., et al. (2025b). [G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks](https://arxiv.org/abs/2410.11782). *ICML 2025*.

[11] [Dynamic Generation of Multi LLM Agents Communication Topologies with Graph Diffusion Models (GTD)](https://arxiv.org/abs/2510.07799). 2025.

[12] [Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](https://arxiv.org/abs/2507.18224). 2025.

[13] Zhang, G., et al. (2025c). [GNNs as Predictors of Agentic Workflow Performances (FLORA-Bench)](https://arxiv.org/abs/2503.11301).

[14] [GLOW: Graph-Language Co-Reasoning for Agentic Workflow Performance Prediction](https://arxiv.org/abs/2512.15751). 2025.

[15] Gu, X., et al. (2024). [Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast](https://arxiv.org/abs/2402.08567). *ICML 2024*.

[16] Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems. 2024.

[17] [TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](https://arxiv.org/abs/2511.05269). 2025.

[18] [Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents](https://openreview.net/forum?id=V4y0CpX4hK). *ICLR 2025*.

[19] Wang, X., et al. (2025a). [OpenHands: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/abs/2407.16741). *ICLR 2025*.

[20] Embrace The Red. (2025a). [OpenHands and the Lethal Trifecta: Leaking Your Agent's Secrets](https://embracethered.com/blog/posts/2025/openhands-the-lethal-trifecta-strikes-again/). 2025.

[21] Embrace The Red. (2025b). [ZombAI Exploit with OpenHands: Prompt Injection to Remote Code Execution](https://embracethered.com/blog/posts/2025/openhands-remote-code-execution-zombai/). 2025.

[22] Invariant Labs. (2025). [Enhancing Browser Agent Safety with Guardrails](https://invariantlabs.ai/blog/enhancing-browser-agent-safety). 2025.

[23] Debenedetti, E., et al. (2025). [Defeating Prompt Injections by Design (CaMeL)](https://arxiv.org/abs/2503.18813). *Google DeepMind*.

[24] Asan, E., et al. (2025). [Design Patterns for Securing LLM Agents against Prompt Injections](https://arxiv.org/abs/2506.08837).

[25] Hua, W., et al. (2024). [TrustAgent: Towards Safe and Trustworthy LLM-based Agents](https://arxiv.org/abs/2402.01586). *EMNLP 2024*.

[26] [VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation](https://arxiv.org/abs/2510.05156). 2025.

[27] AgentSpec: Customizable Runtime Safety for LLM Agents. *ICSE 2026*.

[28] Zhang, H., et al. (2024). [R-Tuning: Instructing Large Language Models to Say 'I Don't Know'](https://arxiv.org/abs/2311.09677). *NAACL 2024 Outstanding Paper*.

[29] [The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235). 2025.

[30] Wang, K., Zhang, G., et al. (2025b). [G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems](https://arxiv.org/abs/2502.11127). *ACL 2025*.

[31] Yu, Y., et al. (2025). [NetSafe: Exploring the Topological Safety of Multi-agent Networks](https://arxiv.org/abs/2410.15686). *ACL 2025*.

[32] [MAMA: Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](https://arxiv.org/abs/2512.04668). 2024.

[33] [Risk-Sensitive Agent Compositions](https://arxiv.org/abs/2506.04632). 2025.

[34] [VeriPlan: Integrating Formal Verification and LLMs into End-User Planning](https://dl.acm.org/doi/10.1145/3706598.3714113). *CHI 2025*.

[35] [Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification](https://arxiv.org/abs/2510.03469). 2025.

[36] [A Survey on Trustworthy LLM Agents: Threats and Countermeasures](https://arxiv.org/abs/2503.09648). 2025.

[37] [Security of LLM-based Agents Regarding Attacks, Defenses, and Applications: A Comprehensive Survey](https://www.sciencedirect.com/science/article/abs/pii/S1566253525010036). *Information Fusion*, 2025.

[38] [Graph-Augmented Large Language Model Agents: Current Progress and Future Prospects](https://arxiv.org/abs/2507.21407). 2025.

[39] [Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities](https://arxiv.org/abs/2506.18019). 2025.
