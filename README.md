# AI Assistant - Task 3: Agentic ReAct Loop & Evaluation System

## Overview
This project extends the W15 production assistant into an agentic system using a single-agent **ReAct (Reasoning + Acting) loop** powered by local **Llama 3.1 (8B)** via **Ollama**. Unlike fixed single-pass pipelines, this assistant dynamically evaluates intermediate search results, decides whether additional tool invocations are necessary, and enforces safety stopping criteria.

> **Feature Justification:** Single-pass RAG pipelines execute fixed retrieval steps regardless of context quality. An agentic loop is necessary here because complex user queries require evaluating whether intermediate tool outputs are sufficient, performing follow-up searches across different tools, and synthesizing multi-step actions dynamically.

---

## 🏗️ System Architecture & Agentic Flow

```text
-----------------------------------------------------------------------------------+
|                                 USER INPUT QUERY                                  |
|         ("Calculate uptime for 10 active days & verify security policy")         |
-----------------------------------------------------------------------------------+
|
v
-----------------------------------------------------------------------------------+
|                          AGENTIC REACT LOOP ENGINE                                |
|        (Iteration Controller - Enforces Max Loop Ceiling: <= 4 Steps)            |
-----------------------------------------------------------------------------------+
|
----------------------+----------------------+
| (Evaluate Evidence & Decides Next Step)    |
v                                             v
------------------------------------+        +------------------------------------+
|            RAG SEARCH              |        |            SYSTEM TOOLS            |
|  - ChromaDB Vector Store           |        |  - Uptime Calculator               |
|  - Ollama Dense Embeddings         |        |  - Security Policy Validator       |
------------------------------------+        +------------------------------------+
|                                             |
----------------------+----------------------+
|
v
-----------------------------------------------------------------------------------+
|                    CONTEXT ENGINEERING: OBSERVATION COMPACTION                    |
|             (Summarizes raw tool responses to prevent token bloat)                 |
-----------------------------------------------------------------------------------+
|
v
-----------------------------------------------------------------------------------+
|                          STOP CONDITION & VALIDATION                              |
|           (Is info complete OR iteration cap reached? -> Return Output)           |
-----------------------------------------------------------------------------------+

```

## 📄 Assignment Documentation Requirements

### a. Context Engineering Technique
1. **Technique Used:** Observation Compaction (`compact_observation`).
2. **Where Applied:** Applied inside the ReAct loop directly after a tool executes and before its result is appended back to the conversation message history (`messages.append`).
3. **Problem Solved:** Long raw vector DB retrievals and verbose system outputs quickly inflate prompt context size during multi-turn loops. Compaction truncates and formats raw outputs into concise summary tokens (`[tool_name Summary]: ...`). This prevents context window saturation and reduces LLM inference latency across iterations.

---

### b. Agentic Pattern
* **Chosen Architecture:** Single-Agent Loop (with ReAct tool binding).
* **Justification:** A single-agent design was selected over a multi-agent structure because the domain tasks (RAG lookups and system policy checks) share a single coherent execution path. 
* **Failure Analysis Alignment:** Introducing a multi-agent supervisor pattern for this scope would introduce unnecessary **Sequential Bottlenecks** and **Context Saturation** due to inter-agent routing overhead. A bounded single-agent loop handles intermediate tool decisions efficiently while avoiding multi-agent coordination costs.

---

### c. Evaluation Harness Results

The evaluation harness (`eval_harness.py`) was built from scratch without external frameworks to benchmark agent performance across task completion, token usage, and error resiliency.

#### Evaluation Metrics Summary Table

| Test ID | Completed | Steps | Tokens Used | Latency | Tool Correctness | Failure Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1 (RAG Retrieval)** | `True` | 4 | 767 | ~2.1s | `True` | **None** |
| **Q2 (Multi-Tool System)** | `False` | 4 | 636 | ~2.4s | `True` | **Soft Failure (Incomplete)** |
| **Q3 (Failure Injection)** | `False` | 1 | 78 | ~0.1s | `N/A` | **Hard Failure (Injected)** |

#### Failure Taxonomy Definitions & Observations
* **None (Q1):** The agent executed `rag_search` across multiple steps and returned a valid response.
* **Soft Failure (Q2):** The agent successfully identified and executed required tools (`calculate_system_uptime`), but reached the hard loop boundary (`max_iterations = 4`) before finalizing its synthesized output.
* **Hard Failure (Q3):** The tool execution engine encountered an unrecoverable exception (`tool_unavailable`). The system immediately halted without hallucinating a response.

---

## 📌 Additional Requirements

### 1. Skill vs. Agent Boundary
> **Analysis:** "Could this feature have been implemented as a Skill?"
> A Skill (loading markdown instructions into context) would have been sufficient for guiding static formatting, but **insufficient** for dynamic multi-step execution. An agent loop was chosen because the execution path (deciding whether to check policy *after* computing uptime) depends entirely on runtime output.

### 2. Token and Cost Accounting
* **Q1 Token Count:** 767 tokens (Single-Agent ReAct)
* **Q2 Token Count:** 636 tokens
* **Q3 Token Count:** 78 tokens
* *Multi-Agent Comparison Note:* Using a multi-agent supervisor model would incur roughly ~1.8× to 2.5× higher token usage (~1,500+ tokens for Q1) due to redundant system prompt distribution and agent-to-agent message passing.

### 3. Failure Injection Test
* **Test Design:** Forced an unhandled exception (`tool_unavailable`) during step 1 of the agent loop.
* **Observed Response:** The agent caught the exception, avoided producing a confident answer based on missing data, and safely returned an explicit failure message to the user: `Agent execution halted due to failure: Tool execution service unreachable`.

### 4. Tool vs. Agent Boundary
* **Design Choice:** External services (ChromaDB vector search and uptime calculations) are modeled as **bounded tool calls** rather than sub-agents.
* **Reasoning:** These services are deterministic, stateless input-output operations. Wrapping them inside autonomous sub-agents would create unnecessary communication overhead without adding decision-making value.

---

## 🚀 Quickstart & How to Run

1. **Activate Virtual Environment:**
   ```powershell
   .venv\Scripts\activate
2. **Run Evaluation Harness:**
    ```powershell
    python eval_harness.py
3. **Launch Streamlit Web Interface:**
    ```powershell
    streamlit run app.py
