# Juliet

**Junctive Unstructured Learning for Incrementally Evolving Transformers**

Juliet is an experimental AI framework built for **incremental, evolutionary learning**.
Unlike static models that freeze after training, Juliet creates a **living ecosystem of isomorphic entities (isos)** — lightweight transformer-based learners that grow, adapt, and evolve over time.

Each iso maintains its own **persistent memory** (via vector stores) and ingests unstructured data from conversations, documents (`.pdf`, `.txt`, etc.), or other sources. Over time, isos fine-tune **lightweight adapters** on their unique history, giving rise to divergent traits, behaviors, and emergent intelligence.

Juliet goes beyond conventional RAG pipelines — it’s a **wild evolutionary soup of transformers**, where memory, state-space dynamics, and perturbations converge to produce creative and adaptive learning.

* disclaimer: Although I do provide context and guidance, most of the comments and docs, including this README, are written by ai. On a more personal note, that has been by far the most impactful quality of life improvement for me. As a disclaimer, it is terrible practice to NOT read every line that your ai inserts into your projects. You are fully responsible for the actions and impacts of ai contributions in your own projects. The truth is, handing your code over to a capable model and asking it to review, comment, and document everything is one of the most underrated things you can do. It’s a disservice to humanity not to take advantage of this. As for project level context, I intend to address, and hopefully solve for that, with this project. This is a living README and will be updated as the project evolves.
---

## Key Concepts

### **Isomorphic Entities (Isos)**

Small transformer instances sharing identical architectures but with **divergent experiences and histories**.

* Each iso evolves through interaction and ingestion.
* Fine-tuned adapters act as localized “epigenetics,” preserving iso-specific traits.

### **Persistent Memory**

Juliet leverages **vector store–based memory** to maintain continuity across sessions.

* Isos don’t just retrieve facts—they *remember* and integrate knowledge into their evolving identity.

### **Incremental Evolution**

Lightweight fine-tuning pipelines build iso-specific adapters.

* Over time, isos differentiate, recombine, or retire — **Darwinian selection in silicon**.

### **State Space Perturbations**

Feed-forward layer perturbations introduce structured “mutations,” enabling:

* Creative leaps and non-linear behavior.
* Divergent reasoning patterns between isos sharing the same base model.

### **Junctive Learning**

Isos don’t evolve in isolation.

* They converse, exchange memories, and fuse knowledge across **junctions**, fostering higher-order intelligence.

---

## Why Juliet?

Juliet isn’t about bigger models—it’s about **richer dynamics** and **emergent behaviors**.

---

## Planned Features

* **Iso Lifecycle:** spawn, evolve, and retire entities.
* **Multi-Format Ingestion:** ingest `.pdf`, `.txt`, and raw conversation streams.
* **Memory Persistence:** store, query, and refine knowledge continuously.
* **Adapter Evolution:** train iso-specific fine-tuned adapters.
* **State Perturbations:** experiment with layer-wise feed-forward perturbations.
* **Iso Junctions:** fuse knowledge across isos through conversation and memory exchange.


## Architecture Overview

Juliet is designed as a **modular ecosystem** of evolving isos, each with its own memory, adapters, and state-space dynamics.

```
           ┌─────────────────────────┐
           │         User Input       │
           └────────────┬────────────┘
                        │
                        ▼
           ┌─────────────────────────┐
           │        Iso Engine        │
           ├────────────┬────────────┤
           │ Iso #1     │ Iso #2     │
           │ (Transformer)             │
           │   ┌───────┐              │
           │   │Adapter│              │
           │   └───┬───┘              │
           │       │                  │
           │   State Space             │
           │   Perturbations           │
           └───────┬──────────────────┘
                   │
                   ▼
           ┌─────────────────────────┐
           │   Persistent Memory      │
           │ (Vector Store / Chroma) │
           └────────────┬────────────┘
                        │
                        ▼
           ┌─────────────────────────┐
           │ Iso Junctions & Fusion   │
           │ Knowledge Exchange       │
           └────────────┬────────────┘
                        │
                        ▼
           ┌─────────────────────────┐
           │   Iso Response Output    │
           └─────────────────────────┘
```

### Components

* **Iso Engine:** Hosts multiple isos. Each iso has a transformer backbone and a lightweight adapter that captures iso-specific evolution.
* **Adapters:** Serve as “epigenetics,” fine-tuned per iso to encode individual experiences.
* **State Space Perturbations:** Feed-forward layer injections that introduce structured mutations, allowing creativity and behavioral divergence.
* **Persistent Memory:** Stores long-term knowledge and conversation history, retrieved dynamically for context.
* **Iso Junctions:** Points where isos exchange memories, merge knowledge, or collaborate, creating higher-order intelligence.

### Workflow

1. User sends input to the system.
2. Each iso processes input through its transformer + adapter.
3. State-space perturbations allow isos to diverge or explore novel responses.
4. Persistent memory is queried to provide context and continuity.
5. Iso junctions enable knowledge fusion across multiple isos.
6. Each iso generates a response that evolves over time as new data is ingested.
