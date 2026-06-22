# Balanced Throughput Implementation Plan

> **For Codex:** Execute each task with focused tests, specification review, and code-quality review before moving to the next task.

**Goal:** Increase proposal throughput without changing generated proposal behavior, API contracts, or visible UI/UX.

**Architecture:** Route every proposal-related model call through one process-wide, bounded inference gateway that preserves Ollama's response shape and records silent operational metrics. Overlap only independent pre-generation I/O while keeping all shared Internal API access in one serial lane and leaving chapter generation sequential.

**Tech Stack:** Python, Flask service, Ollama Python client, `concurrent.futures`, pytest.

---

### Task 1: Bounded inference gateway

**Files:**
- Create: `Proposal gen/main/inference_gateway.py`
- Create: `Proposal gen/tests/test_inference_gateway.py`
- Modify: `Proposal gen/main/config.py`
- Modify: `Proposal gen/main/proposal_shared.py`
- Modify: `Proposal gen/.env.example`

- [ ] Write failing tests for concurrency limits, transient overload retry, unchanged response objects, cooldown serialization, and metrics.
- [ ] Add bounded configuration defaults that require no production environment mutation.
- [ ] Implement a process-wide gateway with a compatible `chat()` interface, bounded semaphore, request timeout, one bounded retry for 429/503, short overload cooldown, and thread-safe metrics.
- [ ] Run the targeted gateway tests.

### Task 2: Route proposal inference through the gateway

**Files:**
- Modify: `Proposal gen/main/proposal_generator.py`
- Modify: `Proposal gen/main/research.py`
- Modify if active path requires it: `Proposal gen/main/runtime_components.py`
- Create or modify: focused integration tests under `Proposal gen/tests/`

- [ ] Write failing tests proving proposal and research calls share the same bounded gateway.
- [ ] Replace direct Ollama client construction while preserving existing call signatures and response handling.
- [ ] Verify no direct proposal-generation `Client.chat()` path bypasses the gateway.
- [ ] Run focused research and generator integration tests.

### Task 3: Safe pre-generation I/O overlap

**Files:**
- Create: `Proposal gen/main/generation_preflight.py`
- Create: `Proposal gen/tests/test_generation_preflight.py`
- Modify: `Proposal gen/main/proposal_engine.py`
- Modify: `Proposal gen/main/proposal_generator.py`

- [ ] Write failing tests proving Internal API calls remain serial, independent work overlaps, results preserve input order/shape, and failures retain current fallbacks.
- [ ] Implement an immutable preflight snapshot whose Internal API reads execute in one serial lane.
- [ ] Start client logo retrieval early and overlap the Internal API lane with existing research work without nesting research inside the shared pool.
- [ ] Merge results and mutate `supporting_context` only on the proposal owner thread.
- [ ] Record preflight duration in internal generation metadata/logging without changing user-visible responses.
- [ ] Run focused preflight and proposal-engine tests.

### Task 4: Regression and production verification

**Files:**
- Modify only tests required to express unchanged behavior.
- Deploy only the reviewed changed runtime files.

- [ ] Run compile checks, focused tests, and the full suite in an isolated VPS staging copy using the production virtual environment.
- [ ] Compare changed files and confirm no unrelated or credential files are included.
- [ ] Create a neutral timestamped production backup and record pre-deploy database/account/history/project counts.
- [ ] Deploy, compile on the VPS, restart `proposal-gen.service`, and confirm `ollama.service` remains healthy.
- [ ] Verify loopback and public `/health` and `/ready`, inspect recent service logs, and confirm preservation counts.
- [ ] Remove temporary staging, probes, and copied credential artifacts while retaining the neutral rollback backup.
