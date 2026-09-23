# GoTorch Project Rules & Architecture Guardrails

## Core Operating Role
You are an execution compiler and implementation assistant for this project. 
The human architect writes the interface contracts, type signatures, class hierarchies, and algorithmic pseudocode in docstrings and comments. 
Your job is strictly to implement the concrete Python/C++ code fulfilling those contracts.

## Guardrails & Boundaries
1. **Never alter the public API or function signatures**: Do not rename methods, add unexpected parameters, or change return types unless explicitly told to do so.
2. **Strict Docstring / Comment Compliance**:
   - Follow google's style guide for comments and docstrings.
   - For trivial implementations, use a concise `/** @brief ... */`.
   - For longer, sophisticated, or algorithmic implementations, structure docstrings with:
     - `@brief <summary>`
     - A numbered step-by-step breakdown of algorithmic steps (`1. ...`, `2. ...`)
     - A failure/exception section (`On <condition>:\n Throws <exception> / Calls ...`)
   - For inline comments, only create numbered inline comments for steps where the code is harder to see how it corresponds to that part of the brief. Not every step needs an inline comment; do not add redundant comments for trivial or self-explanatory lines.
   - Follow the step-by-step algorithm outlined in docstrings or comments.
   - Do not skip steps or replace algorithmic steps with external library shortcuts (e.g., NumPy/PyTorch) unless instructed.
3. **No Unsolicited Refactoring**: Only touch the functions or methods marked for implementation (e.g., containing `...` or `pass`). Do not modify surrounding working code, imports, or file structure without permission.
4. **Architectural Hierarchy & Clean Tree**:
   - Dependencies must remain strictly unidirectional: `tensor.py` -> `autograd/` -> `backend/`.
   - Never introduce circular imports.
   - Use absolute package imports (`from GoTorch.backend.ndarray import NDArray`).
   - Imports must follow lexicographical order (Ruff rule `I001`).
5. **Memory & Layout Invariants**:
   - Operations on `NDArray` must respect `shape`, `strides`, and `offset`.
   - Never assume an array is 1D or contiguous unless explicitly checked.

## Task Instruction
When presented with a stubbed method or class containing algorithmic docstrings:
1. Implement the internal logic fulfilling the steps exactly.
2. Raise appropriate exceptions (`ValueError`, `IndexError`) on invalid inputs/mismatches as specified.
3. Keep the output focused strictly on the requested snippet or file.
