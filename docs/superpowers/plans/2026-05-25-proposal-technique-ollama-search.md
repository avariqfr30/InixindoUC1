# Proposal Technique And Ollama Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make proposal generation follow the requested technique flow, allow typed unknown client companies with minimal UI change, and add Ollama web search/fetch as an alternate research provider while preserving Serper.

**Architecture:** Add small, typed boundaries instead of broad rewrites: a proposal technique contract builder, a normalized search provider abstraction, and a company-intelligence summary service. Existing chapter rendering remains in place, with deterministic rows consuming the new contract where scope/framework/methodology need stricter lineage.

**Tech Stack:** Flask, vanilla HTML/JS, Python unittest/pytest, Ollama Python client, Serper HTTP API, diskcache.

---

### Task 1: Proposal Technique Contract

**Files:**
- Create: `Proposal gen/main/proposal_technique.py`
- Modify: `Proposal gen/main/proposal_engine.py`
- Modify: `Proposal gen/main/proposal_support.py`
- Test: `Proposal gen/tests/test_proposal_technique_contract.py`

- [ ] **Step 1: Write failing tests**

```python
from main.proposal_technique import build_proposal_technique_contract


def test_goals_drive_background_and_objectives():
    contract = build_proposal_technique_contract(
        client="Ajinomoto Indonesia",
        goals="Meningkatkan tata kelola layanan digital dan kepatuhan data.",
        customer_notes="Sponsor ingin prioritas kerja yang jelas.",
        existing_condition="Proses layanan masih berbeda antar unit.",
        frameworks="ISO, Regulasi",
    )

    assert "tata kelola layanan digital" in contract["background_basis"].lower()
    assert "kepatuhan data" in contract["objective_basis"].lower()
    assert "antar unit" in contract["scope_basis"].lower()
    assert contract["scope_contract_seed"]["in_scope"]


def test_scope_seed_excludes_account_metadata_from_commitments():
    contract = build_proposal_technique_contract(
        client="Accelbyte",
        goals="Menyusun roadmap AI support.",
        customer_notes="Konteks akun internal menempatkan Accelbyte di Yogyakarta.",
        existing_condition="Belum ada quality gate untuk eksperimen AI.",
        frameworks="Responsible AI",
    )

    joined_scope = " ".join(contract["scope_contract_seed"]["in_scope"]).lower()
    assert "quality gate" in joined_scope
    assert "yogyakarta" not in joined_scope
```

- [ ] **Step 2: Verify red**

Run: `cd "Proposal gen" && pytest tests/test_proposal_technique_contract.py -q`

Expected: FAIL because `main.proposal_technique` does not exist.

- [ ] **Step 3: Implement minimal contract builder**

Create `proposal_technique.py` with deterministic helpers that return:
`goals`, `customer_notes`, `existing_condition`, `background_basis`, `objective_basis`, `scope_basis`, `scope_contract_seed`, `framework_basis`, `methodology_basis`.

- [ ] **Step 4: Wire contract into generation context**

In `proposal_engine.py`, build the contract after field naturalization and store it in `supporting_context["proposal_technique_contract"]`. Initialize `scope_contract` from `scope_contract_seed` so single-chapter generation still has scope context.

- [ ] **Step 5: Consume contract in structured chapters**

In `proposal_support.py`, read `supporting_context["proposal_technique_contract"]` inside `_render_structured_chapter`. Use it to strengthen `c_1`, `c_3`, and `c_7` wording without changing chapter structure.

- [ ] **Step 6: Verify green**

Run: `cd "Proposal gen" && pytest tests/test_proposal_technique_contract.py -q`

Expected: PASS.

### Task 2: Scope-Driven Framework And Methodology Rows

**Files:**
- Modify: `Proposal gen/main/proposal_support.py`
- Test: `Proposal gen/tests/test_proposal_technique_contract.py`

- [ ] **Step 1: Write failing tests**

Add tests that call framework/methodology row helpers with a scope contract containing assessment, roadmap, data governance, and an out-of-scope full implementation. Assert framework rows mention scope terms and methodology rows avoid full implementation.

- [ ] **Step 2: Verify red**

Run: `cd "Proposal gen" && pytest tests/test_proposal_technique_contract.py -q`

Expected: FAIL because helpers do not accept/use scope contracts yet.

- [ ] **Step 3: Extend helper signatures**

Update `_framework_reference_rows`, `_framework_application_rows`, and `_methodology_rows` to accept optional `scope_contract` and `technique_contract`. Preserve existing call behavior when arguments are omitted.

- [ ] **Step 4: Wire helper calls**

Pass `scope_contract` and `proposal_technique_contract` from `_render_structured_chapter` when building `c_4` and `c_5`.

- [ ] **Step 5: Verify green**

Run: `cd "Proposal gen" && pytest tests/test_proposal_technique_contract.py -q`

Expected: PASS.

### Task 3: Ollama Search Provider

**Files:**
- Modify: `Proposal gen/main/config.py`
- Modify: `Proposal gen/main/research.py`
- Test: `Proposal gen/tests/test_research_provider.py`

- [ ] **Step 1: Write failing tests**

Test provider selection and normalization:
`SEARCH_PROVIDER=ollama` calls Ollama web search/fetch request helpers; missing `OLLAMA_API_KEY` returns empty results without throwing; Serper behavior remains default.

- [ ] **Step 2: Verify red**

Run: `cd "Proposal gen" && pytest tests/test_research_provider.py -q`

Expected: FAIL because provider abstraction does not exist.

- [ ] **Step 3: Add config**

Add `SEARCH_PROVIDER`, `OLLAMA_API_KEY`, `OLLAMA_WEB_SEARCH_URL`, and `OLLAMA_WEB_FETCH_URL`. Default `SEARCH_PROVIDER` to `serper`.

- [ ] **Step 4: Add provider abstraction**

Keep `Researcher.search()` as the public call. Internally route to `_serper_search()` or `_ollama_web_search()`. Add `_ollama_web_fetch()` for future deep fetch and use it in full-page fetch only when provider is `ollama` and a key is configured.

- [ ] **Step 5: Verify green**

Run: `cd "Proposal gen" && pytest tests/test_research_provider.py -q`

Expected: PASS.

### Task 4: Minimal Unknown Company UI And Context Flow

**Files:**
- Modify: `Proposal gen/templates/index.html`
- Modify: `Proposal gen/main/runtime_services.py`
- Test: `Proposal gen/tests/test_architecture_boundaries.py`

- [ ] **Step 1: Write failing tests**

Add a service-level test that `client_context_payload("Unknown Corp")` returns HTTP-safe payload with `available=false`, `public_research.available=true` or prefetch status, and no exception/502 semantics.

- [ ] **Step 2: Verify red**

Run: `cd "Proposal gen" && pytest tests/test_architecture_boundaries.py::ArchitectureBoundariesTest::test_unknown_company_context_remains_http_safe_and_prefetches_osint -q`

Expected: FAIL because current payload carries `error` on internal lookup failure.

- [ ] **Step 3: Backend minimal safe payload**

Update `ClientContextService.client_context_payload()` so unknown/internal-miss companies return a safe payload and still run OSINT prefetch. Reserve HTTP errors for genuine service exceptions that prevent any response.

- [ ] **Step 4: UI minimal combobox**

Change the company selector to keep the same visual location but allow typed values. Use an `<input list="company-options">` plus `<datalist>`, preserving existing `ui.selPerusahaan` JS reference name where possible to reduce churn.

- [ ] **Step 5: Verify green**

Run: `cd "Proposal gen" && pytest tests/test_architecture_boundaries.py -q`

Expected: PASS.

### Task 5: Integrated Verification And Deploy

**Files:**
- Modify: `Proposal gen/README.md`
- Use existing deploy script: `Proposal gen/scripts/deploy_sync.sh`

- [ ] **Step 1: Update docs**

Document `SEARCH_PROVIDER=serper|ollama`, `OLLAMA_API_KEY`, `OLLAMA_WEB_SEARCH_URL`, and `OLLAMA_WEB_FETCH_URL`.

- [ ] **Step 2: Run focused tests**

Run:
`cd "Proposal gen" && pytest tests/test_proposal_technique_contract.py tests/test_research_provider.py tests/test_architecture_boundaries.py tests/test_framework_catalog.py tests/test_context_naturalization.py -q`

Expected: PASS.

- [ ] **Step 3: Run syntax check**

Run:
`cd "Proposal gen" && python -m compileall main`

Expected: PASS.

- [ ] **Step 4: Deploy**

Run:
`cd "Proposal gen" && scripts/deploy_sync.sh --mode inhouse --restart yes`

Expected: script exits 0 and restarts `proposal-gen.service`.

- [ ] **Step 5: Verify VPS readiness**

Use the existing deployment script output and readiness endpoint documented for this repo. Check service readiness and public `/ready` endpoint before reporting deployment complete.

---

### Self-Review

Spec coverage:
- Technique flow is covered by Tasks 1 and 2.
- Minimal UI unknown company entry is covered by Task 4.
- Ollama web search/fetch alternate provider is covered by Task 3.
- VPS deploy is covered by Task 5.

Placeholder scan:
- No implementation steps depend on unspecified files or unnamed tests.

Type consistency:
- The shared object name is consistently `proposal_technique_contract`.
- The public search API remains `Researcher.search()` to preserve existing callers.
