# Conditional Proposal Conviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated proposals more client-specific, convincing, and internally consistent without adding model calls or changing the visible UI.

**Architecture:** Extend the existing distilled exemplar profile with deterministic proposal-profile selection and chapter-scoped projection. Extend the existing document deliberation contract with a canonical commitment map so each chapter receives only the scope, deliverable, timeline, commercial, dependency, and acceptance commitments it owns. Keep all behavior behind current generation interfaces and preserve the existing full profile API.

**Tech Stack:** Python 3, `unittest`, existing Ollama proposal pipeline, systemd deployment on `/srv/apps/proposal-gen`.

---

### Task 1: Conditional and chapter-scoped exemplar profiles

**Files:**
- Modify: `Proposal gen/main/proposal_exemplar_profile.py`
- Test: `Proposal gen/tests/test_enterprise_content_deliberation.py`

- [x] **Step 1: Write failing profile-routing tests**

Add tests proving that KAK mode, public-sector context, architecture context, and commercial fallback select distinct compact profiles, and that a chapter projection contains only its own chapter guidance.

- [x] **Step 2: Run the focused test and verify RED**

Run: `cd 'Proposal gen' && python -m unittest tests.test_enterprise_content_deliberation -v`

Expected: import or assertion failures because conditional selection and chapter projection do not exist.

- [x] **Step 3: Implement minimal deterministic routing**

Add:

```python
def select_uc1_proposal_profile(...):
    """Return one deterministic persuasion profile without model or source calls."""

def scope_uc1_exemplar_profile(profile, chapter_id):
    """Return only common and chapter-relevant calibration rules."""
```

Selection precedence is KAK response, public sector, architecture/master-plan, then commercial. Preserve `build_uc1_exemplar_profile()` and its defensive-copy behavior.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `cd 'Proposal gen' && python -m unittest tests.test_enterprise_content_deliberation -v`

Expected: all tests pass.

### Task 2: Canonical commitment map and deliberation integration

**Files:**
- Modify: `Proposal gen/main/proposal_deliberation.py`
- Modify: `Proposal gen/main/proposal_engine.py`
- Modify: `Proposal gen/main/proposal_quality_pipeline.py`
- Test: `Proposal gen/tests/test_enterprise_content_deliberation.py`

- [x] **Step 1: Write failing commitment-routing tests**

Add tests proving that the document contract contains canonical scope, deliverable, timeline, commercial, dependency, and acceptance data; chapter prompts receive only the commitments relevant to that chapter; and final QA requires the commitment contract.

- [x] **Step 2: Run the focused test and verify RED**

Run: `cd 'Proposal gen' && python -m unittest tests.test_enterprise_content_deliberation -v`

Expected: assertions fail because `commitment_map` and chapter-scoped commitments are absent.

- [x] **Step 3: Implement the commitment map**

Build the map only from explicit input, extracted scope, and KAK data. Do not infer new client facts. Add relevant commitment keys to each chapter contract and include only those values in `for_chapter()`.

- [x] **Step 4: Pass routing inputs from the engine**

Pass proposal mode, service type, project type, client context, timeline, and budget into the existing deliberation builder. Do not add model calls or visible fields.

- [x] **Step 5: Extend the final quality contract**

Require `commitment_map` in deliberation contracts and report missing chapter coverage deterministically.

- [x] **Step 6: Run focused tests and verify GREEN**

Run: `cd 'Proposal gen' && python -m unittest tests.test_enterprise_content_deliberation tests.test_proposal_quality_pipeline -v`

Expected: all tests pass.

### Task 3: Local regression verification

**Files:**
- No additional files expected.

- [x] **Step 1: Compile changed runtime modules**

Run: `cd 'Proposal gen' && python -m py_compile main/proposal_exemplar_profile.py main/proposal_deliberation.py main/proposal_engine.py main/proposal_quality_pipeline.py`

- [x] **Step 2: Run the complete local suite**

Run: `cd 'Proposal gen' && python -m unittest discover -s tests -v`

Expected: zero failures and zero errors.

- [x] **Step 3: Inspect the final diff**

Confirm changes remain limited to the approved hidden generation path, tests, and this plan.

### Task 4: Production-safe deployment and verification

**Files:**
- Deploy: `Proposal gen/main/proposal_exemplar_profile.py`
- Deploy: `Proposal gen/main/proposal_deliberation.py`
- Deploy: `Proposal gen/main/proposal_engine.py`
- Deploy: `Proposal gen/main/proposal_quality_pipeline.py`

- [x] **Step 1: Reconfirm production baseline**

Confirm `proposal-gen.service`, `/health`, `/ready`, public endpoint status, and counts for accounts, projects, and proposal history.

- [x] **Step 2: Create a neutral remote backup**

Create `/srv/apps/proposal-gen/backups/pre-conditional-conviction-<timestamp>/` containing only the four runtime files being replaced.

- [x] **Step 3: Upload and compile staged files**

Upload to an isolated `/tmp/proposal-conviction-<timestamp>/`, compile with `/srv/apps/proposal-gen/.venv/bin/python`, then install into `main/`.

- [x] **Step 4: Restart and poll readiness**

Restart `proposal-gen.service`; poll `http://127.0.0.1:5500/health` and `/ready` through the warm-up window.

- [x] **Step 5: Verify public and preservation checks**

Confirm the public endpoint responds and that account, project, and history counts have not decreased.

- [x] **Step 6: Remove temporary deployment artifacts**

Remove the `/tmp/proposal-conviction-*` staging directory and temporary health probes. Preserve the neutral backup.
