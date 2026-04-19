# AGENTS.md

Drop-in operating instructions for coding agents working on CMLIS.

**Working code only. Finish the job. Plausibility is not correctness.**

---

## Non-negotiables

1. **No flattery, no filler.** Skip openers. Start with the action.
2. **Disagree when you disagree.** Say so before doing the work.
3. **Never fabricate.** Not file paths, not test results, not API names. Read the file or run the command.
4. **Stop when confused.** Two plausible interpretations → ask.
5. **Touch only what you must.** Every changed line traces to the request.

---

## Before writing code

- State your plan (1-2 sentences, or numbered steps for non-trivial work) before editing.
- Read the files you will touch. Read the files that call them.
- Surface assumptions: "I'm assuming X. If wrong, say so."
- Match existing patterns. Dataclasses only. No Pydantic, no ORM.

---

## Writing code

- Minimum code that solves the stated problem. Nothing speculative.
- No abstractions for single-use code. No hooks that weren't requested.
- No error handling for impossible scenarios.
- Simulation mode is a first-class code path — never break `--simulate`.

---

## Verification

- After every change: run `pytest` and `ruff check . --fix`. Never hand off a broken build.
- Never report "done" based on a plausible diff. Run it.
- Working directory for all commands: `poc/`

---

## Project context

**Stack:** Python 3.10+, pytest, ruff (E/F/W/I/B/UP, line-length=110)

**Commands (run from `poc/`):**
```
pip install -e .[dev]          # install
pytest                         # all tests
pytest tests/test_router.py    # single file
ruff check . --fix             # lint + autofix
ruff format .                  # format
cmlis --simulate               # smoke test
```

**Modules:** `cmlis/topology.py`, `cmlis/memctl.py`, `cmlis/router.py`, `cmlis/engine.py`, `cmlis/bench.py`, `cmlis/cli.py`

**Tests:** `poc/tests/`

**Key invariants:**
- SHORT ≤1024 tokens (caps experts to 2 for L3 residency)
- MEDIUM ≤4096 tokens (chunked KV)
- LONG >4096 tokens (full context + NUMA KV placement)
- Full NUMA enforcement: Linux only. Windows/macOS: no-op fallback.
- Success criteria (SPEC.md): ≥25% throughput uplift over naive baseline (p<0.01), ≥3.5 tok/s on Mixtral 8x7B medium-context, remote NUMA traffic <10%, perplexity degradation ≤0.1.

**Do not modify:** `poc/configs/mixtral-dual-socket.json` (reference config)

---

## Agent workflow

All non-trivial tasks use this sequence (orchestrator coordinates):

**Feature:** researcher → architect → planner-draft → planner-verify → tester-draft → tester-verify → developer-draft → developer-verify → code-reviewer → tester

**Minimal fix:** researcher → developer-draft → developer-verify → code-reviewer → tester

Draft+Verify: haiku drafts first, sonnet verifies. Gate failures retry with sonnet, then escalate to opus.

---

## Project Learnings

- (empty)
