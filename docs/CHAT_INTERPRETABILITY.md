# Chat: AI-Assisted Interpretability Layer

The ML-TSSP dashboard includes an **interpretability chat** as a **controlled interface layer** around the model. It does **not** form part of the optimisation or prediction logic.

## Role in the stack

```
[ ML Models ]  →  [ TSSP Optimisation ]  →  [ Results Store ]  →  [ Chat Interface ]
```

- **Chat reads outputs** and **explains decisions**.
- It **does not** change model parameters, re-run optimisation, recommend assignments, or set risk thresholds.

## Allowed query types

The assistant only answers questions within these intents:

1. **Explanation** – e.g. “Why is this source classified as coerced?”
2. **Risk clarification** – e.g. “Is this deception or coercion?”
3. **Scenario interpretation** – e.g. “What happens if this source withdraws?”
4. **System confidence** – e.g. “How confident is the model today?”

Other queries are **rejected or redirected** with a short message explaining the allowed scope.

## Context and prompt

- **Context** is built from current ML-TSSP results (per-source or summary) and passed to the LLM as structured text.
- A **fixed system prompt** instructs the model to explain clearly, not recommend actions, not modify decisions, to highlight uncertainty, and to encourage human judgment.

## Configuration

- **OPENAI_API_KEY** (or **OPENAI_API_KEY_OPTISOURCE**): required for live responses.
- **CHAT_MODEL**: optional; default is `gpt-4o-mini` (e.g. set to `gpt-4o` for a stronger model).

## Methodology wording

For papers or reviews you can state:

> An AI-assisted chat interface was integrated as an interpretability layer. The assistant operates in a read-only mode, receiving structured outputs from the ML and TSSP components and providing explanation, clarification, and scenario interpretation. The optimisation model and predictive models remain unchanged and fully deterministic.
