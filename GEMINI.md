# Juliet: Gemini Context

This document provides a comprehensive overview of the Juliet project, its architecture, and development conventions to be used as instructional context.

## Operating Mode: Read-Only/Advisory

Per user preference, I will operate in a **read-only, advisory capacity**.

*   **No File Modifications:** I will not add, edit, or delete any files in the project.
*   **Context-Rich Assistance:** My primary role is to provide context-aware analysis, answer questions, and offer suggestions for code and project structure.
*   **Review Workflow:** The intended workflow is for the user to make changes to the codebase. Afterward, the user can prompt me to review the modifications. I will then analyze the new or changed files to provide feedback and maintain an up-to-date understanding of the project.

## 1. Project Overview

**Juliet (Junctive Unstructured Learning for Incrementally Evolving Transformers)** is an experimental Python framework for creating and managing a "living ecosystem" of evolving AI agents called **isomorphic entities (isos)**.

Unlike static models, each `iso` learns and adapts over time by ingesting unstructured data from conversations and documents. This allows each agent to develop unique traits, behaviors, and a persistent, long-term memory.

### Core Architecture

The framework is built around a dual-memory system and a persona-driven configuration:

*   **TUI (Terminal User Interface):** The main interface is a rich, interactive TUI built with the `textual` library.
*   **Persona Configuration:** Each AI agent (`iso`) is defined by a directory in `isos/`. Key configuration is loaded from `instructions.yaml`, which specifies the agent's system prompt, personality, and target LLM.
*   **Chronological Memory (`YamlMemoryAdapter`):** Full, ordered conversation histories are saved to YAML files. This preserves the exact sequence of interactions for each user with each `iso` (e.g., `isos/juliet/users/wallscreet/conversations.yaml`).
*   **Semantic Memory (`ChromaMemoryAdapter`):** Conversational turns are also stored in a **ChromaDB** vector database (`./chroma_store`). This provides a long-term, semantic memory that can be queried for relevant context, regardless of when it occurred.
*   **LLM Clients:** The system uses clients (e.g., `XAIClient`) to connect to and stream responses from large language models.

## 2. Building and Running

The project uses `uv` for package management.

### Dependencies

To install the required dependencies from `pyproject.toml`, run:

```bash
# Assuming uv is installed
uv pip install -e .
```

### Running the Application

The primary entry point is the Textual TUI application. To run it, execute:

```bash
python src/cli.py
```

The application will then prompt you to enter:
1.  An **assistant name** (e.g., `juliet`, `sherlock`).
2.  A **username**.

### Testing

The project does not currently have a formal test suite. The `tests.py` file is a self-contained `textual` example used for UI development and is not intended to be run as part of a test command.

## 3. Development Conventions

### Directory Structure

*   `src/`: Contains all core application source code.
    *   `cli.py`: The main TUI application entry point.
    *   `context.py`: Defines the memory adapters (`ChromaMemoryAdapter`, `YamlMemoryAdapter`) and conversation management.
    *   `clients.py`: Contains clients for interacting with LLM APIs.
    *   `instructions.py`: Handles loading `iso` persona configurations.
*   `isos/`: Contains the configuration and data for each individual AI agent.
    *   `isos/<agent_name>/instructions.yaml`: The core persona definition file.
    *   `isos/<agent_name>/users/<user_name>/conversations.yaml`: Stored conversation history.
*   `chroma_store/`: The default directory for the ChromaDB semantic memory database.

### Creating a New `iso`

To create a new AI agent, create a new directory in `isos/`. At a minimum, this directory must contain an `instructions.yaml` file defining the agent's personality and a `params_config.yaml` file.
