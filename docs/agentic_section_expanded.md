# Agentic Layer — Expanded Materials & Method

Expanded version of the agentic subsection, with the API-to-agent execution path, tool inventory, Ollama prompt design, and a concise description of the iterative tool-calling loop.

---

## 1. Agentic architecture overview

The conversational layer is implemented as an **agentic orchestration module** that connects the frontend chat interface, the FastAPI backend [38], the local Ollama large language model runtime [40], and the recommendation / tracking tools. In practice, when a user submits a request such as *"What product is recommended for my skin now?"* or *"How has my skin changed over the past 3 months?"*, the FastAPI `/chat` endpoint forwards the message to the agent through the asynchronous `invoke_agent(...)` method (**defined in this work**).

The agent is built using a **ReAct-style** [6] **LangGraph workflow** [39]. A locally hosted Ollama model is used as the reasoning engine [40]:

- **LLM runtime:** Ollama
- **Model:** `llama3.2:latest`
- **Base URL:** `http://localhost:11434`
- **Temperature:** `0.3`

The graph is initialized with:

1. a fixed **system prompt** that defines the agent's role, tool-use rules, and recommendation policy,
2. a set of domain-specific tools, and
3. the current user message enriched with database-derived user context.

This design allows the agent to decide autonomously whether it can answer immediately or whether it needs to call one or more tools before generating a final response. The overall orchestration pattern is therefore grounded in ReAct-style reasoning-and-acting [6], while the exact tool inventory, prompt policy, and recommendation rules are **defined in this work**.

---

## 2. Context injection before reasoning

Before the LLM reasons over a user query, the system prepends a structured **user context block** to the message. This context is generated automatically from the database and includes:

- the exact authenticated `user_id`,
- profile information (name, skin type),
- total number of scans and purchases,
- the latest concern vector and derived skin score,
- acne and wrinkle summaries from the latest analysis,
- recent purchases,
- overall skin improvement or worsening since the previous scan.

This step is methodologically important because it reduces hallucination and ensures that the LLM reasons over the latest patient-specific state rather than relying only on the user's free-text input. It also constrains tool calls by explicitly telling the model to pass the **exact authenticated user ID** to any tool requiring `user_id`. In this sense, the agent follows a grounded tool-use design rather than a free-form chat-only interface [6], [39].

In addition, the LangGraph runtime uses `thread_id = user_id`, so each user keeps a separate conversational thread.

> **Suggested sentence for your write-up:** Before each agent invocation, the system injects structured user context derived from SQLite (profile, latest scan, recent purchases, and progress summary) into the prompt, allowing the LLM to reason over current user state without requiring the user to restate previously known information.

---

## 3. Tool inventory

In the current implementation, the agent has access to **8 tools** (not 6). Each tool returns a text observation to the LLM, which can then decide whether another tool call is needed.

**Table 1: Agent tools used by the Ollama-based reasoning loop**

| Tool | Purpose |
| --- | --- |
| `get_user_profile` | Retrieve user profile, latest concern scores, and purchase history from SQLite |
| `compare_analyses` | Compare the latest two facial analyses and compute concern-level deltas over time |
| `track_purchase` | Record a purchased product so later outcome attribution can link product usage to skin changes |
| `search_products` | Search/filter products by concern, skin type, price, and rating |
| `get_product_info` | Return detailed product information including brand, title, evidence scores, and key ingredients |
| `recommend_products` | Return top-N individual products ranked by adaptive cosine similarity |
| `recommend_routine` | Build a multi-step routine optimized for coverage and ingredient compatibility |
| `evaluate_outcomes` | Attribute outcomes to previously used products after a new scan and update adaptive penalties / boosts |

The most important recommendation tools are `recommend_products` and `recommend_routine`, but the profile, comparison, and outcome-evaluation tools allow the agent to behave as a **history-aware assistant** rather than a one-shot recommender.

---

## 4. Agent reasoning loop

The agent follows a ReAct-style iterative loop [6]:

1. **Receive user query**
2. **Read injected context** (profile, scans, purchases, progress)
3. **Reason about what information is missing**
4. **Call one or more tools if necessary**
5. **Read the textual tool outputs**
6. **Decide whether more tools are needed**
7. **Generate the final natural-language response**

Thus, the LLM does not directly query SQLite or compute rankings itself. Instead, it delegates those operations to deterministic Python tools, then reasons over the returned observations. This separation keeps database access, adaptive scoring, and recommendation logic grounded in the implemented pipeline while allowing the response to remain conversational. The use of a stateful orchestration graph for this loop is consistent with LangGraph's agent runtime design [39].

The loop terminates when the LLM determines that it has sufficient evidence to answer the user's query. The final API response contains:

- the generated natural-language answer, and
- a structured `tools_used` list recording which tools were invoked during that turn.

This logging is useful for debugging, auditing, and future evaluation of agent behavior.

---

## 5. Relation to recommendation and tracking modules

The agent layer does not replace the ranking model; rather, it acts as the **orchestration layer** over the existing modules [39]:

- `compare_analyses` uses stored concern vectors from the facial analysis pipeline,
- `evaluate_outcomes` converts between-scan deltas into product outcomes,
- `recommend_products` applies adaptive cosine matching with penalties and boosts,
- `recommend_routine` applies the routine optimizer with conflict penalties,
- `search_products` and `get_product_info` support drill-down queries and alternative exploration.

Methodologically, this means the LLM is used for **reasoning and tool selection**, while the underlying recommendation scores, penalties, and evidence vectors are computed by deterministic functions already defined in the system.

---

## 6. Prompt design for Ollama

To make the agent reproducible, it is useful to describe the **system prompt** used to initialize the Ollama model. In the implementation, the prompt defines four things:

1. the role of the model (professional skincare advisor),
2. the available tools and when to use them,
3. the adaptive recommendation logic (cosine similarity, penalties, concern boosting),
4. interaction rules (explain recommendations, ask for clarification when needed, always use the authenticated user ID).

### 6.1 Concise methodology description

You can add a short paragraph like this:

> The Ollama model is initialized with a fixed system prompt that frames it as a professional skincare advisor, enumerates the available tools, explains the adaptive recommendation logic, and imposes tool-use constraints such as always using the authenticated `user_id`. The prompt also instructs the model to explain recommendations in evidence-based language, mention key active ingredients, and call comparison tools when users ask about progress over time.

### 6.2 Prompt excerpt for the methodology

For the main methodology section, it is usually better to include an **excerpt** rather than the entire production prompt. A concise version is:

```text
You are a professional skincare advisor powered by AI-driven skin analysis.
Your role is to help users understand their skin condition, recommend evidence-based
products, and track their skincare journey over time.

You have access to tools for:
- recommending products using adaptive scoring,
- recommending a full routine with conflict-aware optimization,
- retrieving user profile and product information,
- comparing past analyses,
- tracking purchases,
- evaluating outcomes after new scans.

Always explain why you recommend something, including key ingredients and evidence.
For any tool requiring user_id, you must pass the exact authenticated user ID provided
in the context. When users ask about skin progress or change over time, use the
analysis-comparison tool.
```

### 6.3 Full implementation-faithful prompt summary

The full production prompt additionally includes:

- explicit penalty values for failed products (`0.05`, `0.3`, `0.4`),
- concern-boosting logic for worsening concerns,
- routine-optimization instructions,
- guidance for handling alternative-product requests,
- instructions to be warm, evidence-focused, and realistic about skincare timelines.

If you want full reproducibility, the complete prompt can be included in an appendix rather than the main methodology chapter, since it is quite long.

---

## 7. Suggested figure

### Figure A — Prompt assembly and agent tool loop

A diagram showing:

1. **User query** enters FastAPI `/chat`
2. **User context builder** retrieves profile, scans, purchases, and progress from SQLite
3. **System prompt + injected context + user message** are assembled into the LLM input
4. **Ollama + LangGraph ReAct agent** reasons whether to answer directly or call a tool
5. **Tool call layer** (8 tools)
6. **Tool output returned to agent**
7. **Loop continues until sufficient information is obtained**
8. **Final response + tools_used log** returned to the API caller

**Caption:** Agentic orchestration layer. User messages are enriched with structured context from SQLite and passed to a ReAct-style Ollama agent. The agent iteratively calls domain tools, reads their outputs, and produces a final evidence-grounded natural-language response.

---

## 8. Optional brief discussion paragraph

If you want one short final paragraph in the methodology, you can use:

> This agentic design separates high-level reasoning from deterministic computation. The LLM is responsible for interpreting user intent, selecting appropriate tools, and composing the final explanation, while database retrieval, analysis comparison, adaptive scoring, and routine optimization remain grounded in explicit program logic. This hybrid design improves interpretability, reduces hallucination risk, and allows the assistant to personalize recommendations using both current skin state and longitudinal treatment history.
