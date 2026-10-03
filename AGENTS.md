# GoTorch Project Rules & Architecture Guardrails

## Core Operating Role
You are an execution compiler and implementation assistant for this project. 
The human architect writes the interface contracts, type signatures, class hierarchies, and algorithmic pseudocode in docstrings and comments. 
Your job is strictly to implement the concrete Python/C++ code fulfilling those contracts.

## Guardrails & Boundaries
1. **Never alter the public API or function signatures**: Do not rename methods, add unexpected parameters, or change return types unless explicitly told to do so.
2. **Strict Docstring / Comment Compliance**:
   - Follow Google's style guide for comments and docstrings.
   - For trivial implementations, use a concise `/** @brief ... */`.
   - For longer, sophisticated, or algorithmic implementations, document parameters, return values, and exceptions clearly (`@param`, `@return`, `@throws`).
   - Do not artificially split functions into excessive numbered pseudo-steps.
   - Keep functions small and focused: decompose complex or repetitive logic into small, descriptive helper functions.
   - For inline comments, only create comments where the rationale or non-obvious invariant is needed; do not add redundant comments for self-explanatory lines.
   - Follow the algorithmic specifications outlined in docstrings or comments.
   - Do not skip steps or replace algorithmic steps with external library shortcuts (e.g., NumPy/PyTorch) unless instructed.
3. **No Unsolicited Refactoring**: Only touch the functions or methods marked for implementation (e.g., containing `...` or `pass`). Do not modify surrounding working code, imports, or file structure without permission.
4. **Architectural Hierarchy & Clean Tree**:
   - Dependencies must remain strictly unidirectional: `tensor.py` -> `autograd/` -> `backend/`.
   - Never introduce circular imports.
   - Use absolute package imports (`from GoTorch.backend.native_backend import NDArray`).
   - Imports must follow lexicographical order (Ruff rule `I001`).
5. **Memory & Layout Invariants**:
   - Operations on `NDArray` must respect `shape`, `strides`, and `offset`.
   - Never assume an array is 1D or contiguous unless explicitly checked.

## Task Instruction
When presented with a stubbed method or class containing algorithmic docstrings:
1. Implement the internal logic fulfilling the steps exactly.
2. Raise appropriate exceptions (`ValueError`, `IndexError`) on invalid inputs/mismatches as specified.
3. Keep the output focused strictly on the requested snippet or file.

## Docstring & API Contract Invariants
All C++ and Python functions must include Google-style docstrings:
1. **C++ (`.hpp`, `.cpp`)**:
   - Use `/** ... */` blocks.
   - Must contain: `@brief`, `@param` (for each argument), `@return`, and `@throw` (if exceptions are raised).
2. **Python / Stubs (`.py`, `.pyi`)**:
   - Use triple quotes `"""`.
   - Must contain: One-line summary, `Args:`, `Returns:`, and `Raises:` (if exceptions can occur).
3. **Validation**: Run `make lint` before declaring any task complete. Zero warnings allowed.

## Automation & Testing Workflow
Whenever any file in `GoTorch/` is modified:
1. **Lifecycle Hook Execution**: The `PostToolUse` hook in `.agents/hooks.json` automatically:
   - Runs `make build` if any C++ backend source in `GoTorch/backend/cppsrc/` was touched.
   - Runs `stubgen` (`make stubs`) to regenerate interface stubs under `tests/stubs/GoTorch/`.
2. **Subagent Delegation (Review-First)**:
   - The primary agent MUST delegate test creation to `test-agent` via `invoke_subagent` / `send_message`.
   - The delegation prompt MUST instruct `test-agent` to:
     - Inspect the updated contracts in `tests/stubs/GoTorch/` for the modified module(s).
     - Formulate new or updated tests purely from the interface stubs.
     - **NEVER write or edit files on disk directly**: Return the exact test code, diffs, and explanations strictly in its response message back to the primary agent.
3. **Human Review & Application**:
   - The primary agent presents the proposed test code and diffs to the human developer.
   - The primary agent applies the changes via tool edit requests (`write_to_file` / `replace_file_content`), ensuring full diff visibility and approval directly in the primary terminal.
4. **Verification & Test Failures (User Debugging)**:
   - The primary agent executes targeted linters (`make lint`) and tests (`pytest` / `make test-cpp`) and related integration tests.
   - **If any test fails, STOP immediately**: The human developer wants to debug test failures themselves. Do NOT attempt to iterate, auto-fix, or modify code after a failure. Report the failure, exact command to reproduce, and output to the user.
