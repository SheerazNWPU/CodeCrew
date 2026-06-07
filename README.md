# 💻 CodeCrew

**Multi-Agent Code Generation System** - 6 specialized agents working together to write, test, review, and explain code.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 📋 Overview

**CodeCrew** is a **pure multi-agent system** (no RAG) that demonstrates how specialized agents can collaborate to generate high-quality code. Each agent has a specific role in the code generation pipeline, working together to produce complete, tested, and well-documented code.

### What it demonstrates:
- ✅ **Multi-Agent Architecture** - 6 specialized agents in a sequential pipeline
- ✅ **Code Generation** - Writes Python code from natural language descriptions
- ✅ **Test Generation** - Automatically creates test cases
- ✅ **Code Review** - Reviews and improves code quality
- ✅ **Explanation** - Provides clear, user-friendly explanations

---

## 🏗️ Architecture

### Complete System Architecture

<img src="codecrew_architecture.png" width="100%">

*Figure 1: Complete CodeCrew multi-agent architecture*

### Agent Sequence Diagram

<img src="codecrew_sequence.png" width="100%">

*Figure 2: Agent interaction sequence diagram*

### Simplified Pipeline

<img src="codecrew_flow.png" width="100%">

*Figure 3: Simplified view of the agent pipeline*

### Agent Roles & Responsibilities

<img src="codecrew_agent_roles.png" width="100%">

*Figure 4: Each agent's role and responsibilities*

---

## 🤖 Agent Pipeline

| Step | Agent | Icon | Role | Output |
|------|-------|------|------|--------|
| 1 | Requirements Analyzer | 📋 | Breaks down the coding task | Clear requirements list |
| 2 | Designer | 🏗️ | Designs code architecture | Code structure + approach |
| 3 | Coder | 💻 | Writes the actual code | Complete Python code |
| 4 | Tester | 🧪 | Generates test cases | Test suite |
| 5 | Reviewer | ✅ | Reviews and improves code | Optimized code |
| 6 | Explainer | 📖 | Explains code to user | User-friendly explanation |

### Agent Details

**📋 Requirements Analyzer**
- **Input:** "Write a function to calculate factorial"
- **Output:** Core functionality, edge cases, input/output specifications

**🏗️ Designer**
- **Input:** Requirements
- **Output:** Algorithm choice, code structure, data flow

**💻 Coder**
- **Input:** Design
- **Output:** Complete Python code with docstrings and error handling

**🧪 Tester**
- **Input:** Code
- **Output:** Unit tests, edge case tests, error handling tests

**✅ Reviewer**
- **Input:** Code
- **Output:** Quality feedback, improvements, approval

**📖 Explainer**
- **Input:** Code
- **Output:** User-friendly explanation and usage examples

---

## 📦 Pre-Requisites

### 1. Downloaded Model Checkpoint
- DeepSeek Chat (7B) model (or any HuggingFace compatible LLM)
- Path: `/path/to/your/model/`

### 2. Python Dependencies
```bash
pip install torch transformers
