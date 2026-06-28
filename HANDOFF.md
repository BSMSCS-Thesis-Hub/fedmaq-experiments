# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                             |
| ---------------------- | ------------------------------------------------- |
| **Last updated**       | 2026-06-28                                        |
| **Last session focus** | SOTA baseline alignment and dependency resolution |
| **Active repo**        | fedmaq-experiments                                |
| **Blockers**           | None                                              |

---

## 1. Quick start (new agent)

1. Open the **multi-root workspace** with all five `fedmaq-*` repos.
2. Read this file end-to-end, then the **active task** in [Section 6](#6-implementation-queue).
3. Load domain rules from [`fedmaq-experiments/.cursor/rules/`](.cursor/rules/) (canonical thesis context).
4. Work in **one primary repo** per task; use sibling `AGENTS.md` for entrypoints.
5. Before ending a session, run the **`agent-handoff` skill** (`.cursor/skills/agent-handoff/`) to update this file and recommend whether to hand off for clean context.

**Candidate:** Christian Joseph Bunyi | **Institution:** De La Salle University | **Advisor:** Fritz Flores

---

## 2. Workspace map

| Repo                                             | Role                              | Agent entry                                    | Domain rules                       |
| ------------------------------------------------ | --------------------------------- | ---------------------------------------------- | ---------------------------------- |
| [fedmaq-experiments](../fedmaq-experiments/)     | FedMAQ code, Hydra, Flower, WandB | [AGENTS.md](../fedmaq-experiments/AGENTS.md)   | **Owner:** `.cursor/rules/`        |
| [fedmaq-literature](../fedmaq-literature/)       | PDFs, RAG, summaries              | [AGENTS.md](../fedmaq-literature/AGENTS.md)    | `thesis-context.mdc` → experiments |
| [fedmaq-analyses](../fedmaq-analyses/)           | Notebooks, thesis figures         | [AGENTS.md](../fedmaq-analyses/AGENTS.md)      | `thesis-context.mdc` → experiments |
| [fedmaq-manuscript](../fedmaq-manuscript/)       | LaTeX thesis (template pending)   | [README.md](../fedmaq-manuscript/README.md)    | Defer until template               |
| [fedmaq-presentations](../fedmaq-presentations/) | Beamer slides                     | [AGENTS.md](../fedmaq-presentations/AGENTS.md) | `thesis-context.mdc` → experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content. Index via `../fedmaq-experiments/.cursor/rules/`.

**Cursor layout:** Each repo owns `.cursor/rules/`, `.cursor/skills/`, `.cursor/project/`. No shared parent `.cursor/` (may add later).

---

## 3. Locked architectural decisions

| Topic              | Decision                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| Thesis context     | `fedmaq-experiments/.cursor/rules/` (decomposed from `context.md`; `context.md` is human snapshot only) |
| Experiments layout | uv monorepo, 4 phases under `src/fedmaq/`, baselines in `src/fedmaq/baselines/`                         |
| Tooling            | Preferred stack in `tech-stack.mdc`; adopt extra libs (pandas, sklearn, etc.) when justified            |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                         |
| RAG boundaries     | Drafts → `*/drafts/`; human `approve` before promotion; no cross-repo auto-edits                        |
| Embeddings         | **`Qwen/Qwen3-Embedding-4B`** local GPU; serialize GPU jobs (convert then embed)                        |
| LLM                | OpenRouter: `deepseek/deepseek-v4-flash` (default), `deepseek/deepseek-v4-pro` (synthesis)              |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                          |

---

## 4. Per-repo status

### fedmaq-experiments — [Phase 1 Env Complete]

| Done                                                           | Pending                                         |
| -------------------------------------------------------------- | ----------------------------------------------- |
| `pyproject.toml`, `src/fedmaq/` phase packages, `conf/`, tests | Port baseline code into `src/fedmaq/baselines/` |
| 11 `.cursor/rules/`, registries, 2 skills                      | Docker integration                              |
| `context.md` deprecation notice                                |                                                 |
| Phase 1 environment: model factory, partitioning, caching,     |                                                 |
| telemetry, client/strategy wrappers, `scripts/run.py`          |                                                 |

Key paths: `src/fedmaq/phase1_env/` … `phase4_benchmark/`, `.cursor/project/baseline_registry.md`

### fedmaq-literature — [Complete]

| Done                                                                   | Pending |
| ---------------------------------------------------------------------- | ------- |
| Folder layout, `.cursor/` rules/skills, `paper_registry.md` (complete) | None    |
| Docling + Marker convert pipeline, QA, `meta.yaml`, CLI                |         |
| `fedmaq-lit convert` / `ingest --convert-only`, unit tests             |         |
| Smoke-tested on `hinton-2015-distillation`, `li-2020-fedprox`          |         |
| Batch conversion CLI (`--all` flag) and registration of all 29 papers  |         |
| LlamaIndex + Chroma ingest with Qwen3-4B                               |         |
| OpenRouter summarize & approve workflow CLI                            |         |
| Chroma RAG local query & LLM synthesis CLI                             |         |

Stack: Docling primary, Marker GPU fallback → `markdown/{slug}/` → Qwen3-4B → Chroma → query/summarize.

### fedmaq-analyses — [Scaffold complete]

| Done                                                      | Pending                                 |
| --------------------------------------------------------- | --------------------------------------- |
| `data/` layout + README, plot style stub, sample notebook | WandB/Hydra ingest implementations      |
| `.cursor/` rules, skills, `figure_registry.md`            | Real ablation + thesis figure notebooks |

### fedmaq-presentations — [Complete]

| Done                                                 | Pending |
| ---------------------------------------------------- | ------- |
| `.agents/` → `.cursor/`, metadata aligned to FedMAQ  | None    |
| `slide_registry.md` paths fixed                      |         |
| Slide content updates for vision-FL framing (y3t3w7) |         |

### fedmaq-manuscript — [Active]

| Done                                                           | Pending                                          |
| -------------------------------------------------------------- | ------------------------------------------------ |
| LaTeX template integrated with Chapters 1--4                   | Draft final Chapters 5 and 6                     |
| Granular, non-overlapping Gantt Chart of Activities configured | Incorporate proposal panel feedback post-defense |
| `.cursor/` rules configured (`thesis-context`, `latex_rules`)  |                                                  |

---

## 5. Literature RAG reference (implementation spec)

```txt
papers/*.pdf
  → Docling convert → QA → Marker fallback if low confidence
  → markdown/{slug}/paper.md + meta.yaml
  → LlamaIndex IngestionPipeline → storage/chroma/ (gitignored)
  → fedmaq-lit summarize → summaries/drafts/{slug}.md
  → human: fedmaq-lit approve → summaries/{slug}.md
  → syntheses/drafts/ → approve-synthesis → syntheses/{topic}.md
```

**Embedding:** `FEDMAQ_EMBED_MODEL=Qwen/Qwen3-Embedding-4B`, fallback `0.6B`, batch size 4–8. Query instruct in `src/fedmaq_literature/ingest/__init__.py`.

**LLM Models (OpenRouter):** Use `deepseek/deepseek-v4-flash` for drafting summaries. Always use `deepseek/deepseek-v4-pro` for automated reviews, verification runs, and global thematic syntheses to ensure mathematical correctness.

**GPU (RTX 5060 8GB):** Do not run Docling/Marker and 4B embedder concurrently.

> [!IMPORTANT]
> **Expected Execution Runtimes:**
>
> - **Full PDF to Markdown Conversion (Docling + Marker QA):** ~6.5 to 7 hours total. Avoid re-running conversions from scratch unless necessary.
> - **Full RAG Ingestion & Embedding (Qwen3-Embedding-4B):** ~14 minutes per paper on CUDA GPU (RTX 5060) (total ~6.8 hours for 29 papers). Ingestion is run sequentially in a loop to provide real-time visibility, log updates, and incremental SQLite commits.
> - **Paper Summarization (DeepSeek-v4-Flash via OpenRouter):** ~20 to 25 seconds per paper.

**Skills:** `.cursor/skills/ingest-paper`, `summarize-paper`, `approve-summary`, `query-literature` (synthesize skills TBD).

**Component Roles & Purpose:**

- **Chroma Vector DB (RAG):** Manages split text chunks for granular, passage-level retrieval (e.g., retrieving exact formulas or parameters).
- **Paper Summaries (`summaries/`):** Lightweight markdown files designed to fit easily inside the LLM's context window. They provide high-level summaries of methodology, limitations, and relevance for global reasoning.
- **Thematic Syntheses (`syntheses/`):** Aggregates summaries by topic to trace cross-paper evidence, support core claims, and identify literature gaps.

---

## 6. Implementation queue

Priority order for upcoming work. Mark items `[x]` when done; add new items at the bottom with date.

| P   | Task                                                       | Repo        | Status                  |
| --- | ---------------------------------------------------------- | ----------- | ----------------------- |
| 1   | Implement PDF convert (Docling + Marker QA)                | literature  | [x]                     |
| 2   | LlamaIndex + Chroma ingest with Qwen3-4B                   | literature  | [x]                     |
| 3   | `fedmaq-lit` summarize + approve + OpenRouter              | literature  | [x]                     |
| 4   | Phase 1 FL environment (data partition, bandwidth, Flower) | experiments | [x]                     |
| 5   | FedAvg baseline in `src/fedmaq/baselines/`                 | experiments | [ ]                     |
| 6   | WandB + Hydra ingest utilities                             | analyses    | [ ]                     |
| 7   | Manuscript `.cursor/` stub                                 | manuscript  | [ ] (blocked: template) |
| 8   | Review & approve remaining 10 draft summaries (remediate)  | literature  | [x]                     |
| 9   | Compile/synthesize summaries into thematic syntheses       | literature  | [ ]                     |

> [!TIP]
> For **Task 8**, the agent should perform the corrections locally by reading the critique files (`summaries/drafts/*_critique.md`) and modifying the draft summaries directly, rather than calling OpenRouter APIs. This keeps the workflow fast and cost-free for the user's OpenRouter account.

**Current focus:** P5 — FedAvg baseline in `src/fedmaq/baselines/` (`fedmaq-experiments`).

---

## 7. Environment and secrets

| Variable                      | Used by               | Notes                                         |
| ----------------------------- | --------------------- | --------------------------------------------- |
| `OPENROUTER_API_KEY`          | literature            | LLM via OpenAI-compatible API                 |
| `FEDMAQ_EMBED_MODEL`          | literature            | Default `Qwen/Qwen3-Embedding-4B`             |
| `FEDMAQ_EMBED_FALLBACK_MODEL` | literature            | Default `Qwen/Qwen3-Embedding-0.6B`           |
| `FEDMAQ_EMBED_BATCH_SIZE`     | literature            | Default `4`                                   |
| `FEDMAQ_QA_MIN_MEAN_GRADE`    | literature            | Default `good` (Docling mean_grade threshold) |
| `FEDMAQ_QA_MIN_LOW_GRADE`     | literature            | Default `fair`                                |
| `FEDMAQ_MARKER_DEVICE`        | literature            | Override Marker device (`cuda` / `cpu`)       |
| `HF_HUB_DISABLE_SYMLINKS`     | literature            | Set automatically on Windows in Docling path  |
| `WANDB_API_KEY`               | experiments, analyses | Experiment tracking                           |

Create `.env` locally (gitignored); document new vars here when added.

**Setup:** `uv sync` in each Python repo; `uv sync --extra dev` for pytest in experiments.

---

## 8. Agent conventions

- **Registries** (`.cursor/project/*.md`): update when adding baselines, papers, figures, or runs.
- **Rules** (`.cursor/rules/*.mdc`): concise; one concern per file; experiments owns domain rules.
- **Skills** (`.cursor/skills/*/SKILL.md`): procedural workflows; prefer skills over ad-hoc commands.
- **No emojis** in repo files (`repo-preferences.mdc`).
- **MCP:** context7 (library docs), GitHub (issues/PRs) — user-level.

---

## 9. What not to do

- Parse `papers/*.pdf` directly in Cursor chat.
- Auto-promote LLM drafts to `summaries/` or `syntheses/` without human approve.
- Let RAG agents edit experiments/analyses/manuscript code directly.
- Duplicate thesis domain content outside `fedmaq-experiments/.cursor/rules/`.
- Add top-level `reproductions/` packages; use `src/fedmaq/baselines/`.

---

## 10. Changelog

### 2026-06-28 — SOTA baseline alignment and dependency resolution

- Re-aligned the target baseline suite to match the revised manuscript (Chapters 1–4): Seminal Controls (FedAvg, FedProx), Pure Quantization (FedPAQ, DAdaQuant), Pure KD (FedMD, FedDistill), and Hybrid Q+KD (FedKD, CFD).
- Excluded unreleased/non-reproducible conceptual competitors (DynFed, FedDT, AdaDQ-KD, LAQ-HC).
- Deleted obsolete configuration files for `dynfed`, `feddt`, and `laq_hc` under `conf/algorithm/`.
- Created new algorithm configurations for `fedpaq.yaml`, `feddistill.yaml`, and `cfd.yaml`.
- Integrated `torch` and `torchvision` dependencies (using CUDA 13.2 wheels index) in `pyproject.toml` to fix unit test collection, executing `uv sync` to update the lock file.
- Updated baseline and paper registries in both `experiments` and `literature` workspaces.
- Verified clean execution of the test suite via `pytest`.

### 2026-06-25 — Baselines narrowed to 8 active SOTA algorithms

- Streamlined the thesis scope by narrowing down from 11 baselines to 8 active baselines: FedAvg, FedProx, DAdaQuant, LAQ-HC, FedMD, FedKD, DynFed, and FedDT.
- Set aside FedPAQ, AdaGQ, FedDistill, and AdaDQ-KD for future consideration.
- Created `conf/algorithm/fedmd.yaml` and deleted configuration files for the 4 excluded algorithms.
- Updated baseline registries under `.cursor/project/baseline_registry.md` and `baseline_reference_benchmarks.md`.
- Modified `chapter_4.tex` in the manuscript repository to update the baseline count and table of hyperparameters.
- Successfully verified that the LaTeX manuscript compiles clean and all unit tests continue to pass.

### 2026-06-24 — Phase 1 Federated Learning Environment Completed and Verified

- Implemented standard models (`SimpleCNN`, `ResNet18GN`) and parameter extraction/injection helpers in `fedmaq.core.models`.
- Built Dirichlet data partitioner with determinism and client partitioning cache under `.data_partitions/` in `fedmaq.core.partitioning`.
- Created customized `TelemetryFedAvg` strategy that simulates client upload/download bandwidth and compute speeds to estimate physical wall-clock time in `fedmaq.core.strategy`.
- Robustified configuration parsing across the telemetry, client, and strategy classes to seamlessly support nested Hydra experiment keys and command line overrides.
- Verified and passed all 5 pytest unit tests under `tests/`.
- Verified end-to-end 1-round CPU simulation using the runner script `scripts/run.py`.

### 2026-06-23 — Y3T3W7 slide deck finalization and presentation styling rules update

- Finalized the Beamer slide deck for progress update `y3t3w7` under `updates/y3t3w7/main.tex`.
- Pivoted slide content from IoT smart campus energy forecasting to classical Federated Learning (image classification) and the FedMAQ architecture.
- Integrated SOTA comparison tables, evaluation stack blocks, experimental setups, and manuscript progress checklists.
- Configured margins globally to `1.2cm` in `preamble/packages.tex`, updated documentclass font size from `10pt` to `11pt` in all slide drivers, and updated `.cursor/rules/beamer_rules.mdc` to document this new standard.
- Verified successful LaTeX compilation using `latexmk` with zero overfull horizontal boxes or errors.

### 2026-06-23 — Manuscript Gantt Chart Refinement and Cursor rules initialization

- Re-designed the calendar of activities in `fedmaq-manuscript`'s `chapter_4.tex` to be strictly sequential and non-overlapping.
- Extended the schedule to April 2027 to de-risk baseline implementation and benchmarking under a 15-unit coursework load.
- Completely excluded December 2026 from active research, indicating coursework finals and holidays, and added a justifying text paragraph.
- Configured `.cursor/rules/` stubs in the `fedmaq-manuscript` repository (`thesis-context.mdc`, `latex_rules.mdc`, and `repo-preferences.mdc`).
- Swapped slides preparation and proposal defense/revision rows to match an April 2027 defense timeline.
- Verified successful LaTeX compilation using `pdflatex main.tex`.

---

## 11. Handoff recommendation

When ending a session, the `agent-handoff` skill indicates whether you should hand off to a new agent session for clean context. Do not recommend handoff without updating Section 6 and the changelog.
