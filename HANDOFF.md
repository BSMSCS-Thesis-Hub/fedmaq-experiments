# FedMAQ Workspace Handoff

Living document for agent-to-agent and session-to-session continuity across the FedMAQ thesis multi-repo workspace.

| Field                  | Value                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Last updated**       | 2026-07-02                                                                                                                   |
| **Last session focus** | Implement mathematically fair decoupled client/server simulated runtimes, resolve test flakiness, and run 5-round benchmarks |
| **Active repo**        | fedmaq-experiments                                                                                                           |
| **Blockers**           | None                                                                                                                         |

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
| Experiments layout | uv monorepo, code under `src/fedmaq/core/` and `src/fedmaq/baselines/`                                  |
| Tooling            | Preferred stack in `tech-stack.mdc`; adopt extra libs (pandas, sklearn, etc.) when justified            |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                         |
| RAG boundaries     | Drafts → `*/drafts/`; human `approve` before promotion; no cross-repo auto-edits                        |
| Embeddings         | **`Qwen/Qwen3-Embedding-4B`** local GPU; serialize GPU jobs (convert then embed)                        |
| LLM                | OpenRouter: `deepseek/deepseek-v4-flash` (default), `deepseek/deepseek-v4-pro` (synthesis)              |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                          |

---

## 4. Per-repo status

### fedmaq-experiments — [Phase 1 Env Complete; Manuscript Aligned]

| Done                                                           | Pending                       |
| -------------------------------------------------------------- | ----------------------------- |
| `pyproject.toml`, codebase structure, `conf/`, tests           | Port remaining SOTA baselines |
| 11 `.cursor/rules/`, registries, 2 skills                      | (FedDistill, CFD — Sep 2026)  |
| `context.md` deprecation notice                                | Docker integration            |
| Phase 1 environment: model factory, partitioning, caching,     |                               |
| telemetry, client/strategy wrappers, `scripts/run.py`          |                               |
| Seminal controls (FedAvg, FedProx), pure quantization (FedPAQ, |                               |
| DAdaQuant), model distillation (FedMD), hybrid Q+KD (FedKD),   |                               |
| and revised FedMAQ implementations                             |                               |
| Full manuscript audit (Ch. 1--4); all discrepancies resolved   |                               |
| Bug fixes: seeded RNG for DAdaQuant, heterogeneous comp speed  |                               |
| Config hardening: ci.yaml, kd_epochs, uniform_memory.yaml      |                               |
| Decoupled simulated client/server times & telemetry logger     |                               |
| Mathematical fairness client penalties & server delays added   |                               |
| 5-round GPU benchmark runs executed and analyzed for 7 algos   |                               |

Key paths: `src/fedmaq/core/`, `src/fedmaq/baselines/`, `.cursor/project/baseline_registry.md`

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
| Updated Chapter 4 with decoupled simulated time formulation    |                                                  |
| Added mathematical fairness explanations for client/server KD  |                                                  |

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

| P   | Task                                                       | Repo        | Status |
| --- | ---------------------------------------------------------- | ----------- | ------ |
| 1   | Implement PDF convert (Docling + Marker QA)                | literature  | [x]    |
| 2   | LlamaIndex + Chroma ingest with Qwen3-4B                   | literature  | [x]    |
| 3   | `fedmaq-lit` summarize + approve + OpenRouter              | literature  | [x]    |
| 4   | Phase 1 FL environment (data partition, bandwidth, Flower) | experiments | [x]    |
| 5   | FedAvg / FedProx / FedPAQ / DAdaQuant baselines            | experiments | [x]    |
| 6   | Full manuscript audit + codebase hardening (Ch. 1--4)      | experiments | [x]    |
| 7   | WandB + Hydra ingest utilities                             | analyses    | [ ]    |
| 8   | Review & approve remaining 10 draft summaries (remediate)  | literature  | [x]    |
| 9   | Compile/synthesize summaries into thematic syntheses       | literature  | [ ]    |
| 10  | Port FedDistill baseline                                   | experiments | [ ]    |
| 11  | Port CFD baseline                                          | experiments | [ ]    |

> [!TIP]
> For **Task 8**, the agent should perform the corrections locally by reading the critique files (`summaries/drafts/*_critique.md`) and modifying the draft summaries directly, rather than calling OpenRouter APIs. This keeps the workflow fast and cost-free for the user's OpenRouter account.

**Current focus:** P7 — WandB + Hydra ingest utilities (`fedmaq-analyses`). Tasks 10--11 (FedDistill, CFD) are Sep--Oct 2026 per Gantt.

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

### 2026-07-02 — Uniform System Simulation and Preliminary Config Updates

- **Uniform System Parameters:** Removed heterogeneous bandwidth and compute parameters (from `conf/experiment/default.yaml` and `tests/test_environment.py`). Migrated to uniform system simulation using configurable `bandwidth_mbps` (default 10.0 Mbps) and `compute_samples_per_sec` (default 200 samples/sec) across all clients in `TelemetryFedAvg`.
- **Preliminary Test Setup:** Configured `num_clients: 50` and `total_rounds: 10` for preliminary iterative comparisons.
- **Decoupled simulated runtime:** Implemented decoupled simulated client training time and server processing/distillation time in `TelemetryFedAvg` strategy (`strategy.py`) and `TelemetryManager` (`telemetry.py`), writing them to separate metrics fields (`system/client_sim_time_sec` and `system/server_sim_time_sec`).
- **FedKD compute scaling:** Applied a $2.5\times$ computational penalty scale factor to client compute speeds for `fedkd` to model the increased overhead of training both student and teacher models concurrently.
- **FedMD pre-training delay:** Adjusted the client training delay simulation for `fedmd` during Round 1 to include the 10 public pre-training and 10 private pre-training epochs.
- **FedMAQ server delay:** Formulated and added the server-side proxy ensemble distillation delay ($T_{server}$) for `fedmaq` based on public proxy dataset size, epoch count, and active teachers.
- **Test suite flakiness resolved:** Updated the `get_properties` mock in `test_dadaquant_strategy_allocation` inside `test_environment.py` to deterministically map clients to partitions, preventing hash-seed random fallback failures.
- **5-Round benchmark simulation:** Executed full 5-round MNIST training runs for all 7 FL algorithms (`fedavg`, `fedprox`, `fedpaq`, `dadaquant`, `fedmd`, `fedkd`, and `fedmaq`) using GPU training, validating stable convergence and correct logging.
- **Manuscript alignment:** Updated subsubsection items in `chapter_4.tex` to clean up brackets and synchronize the decoupled simulated time formulation.

### 2026-07-01 — Manuscript Table 4.1 Hyperparameter Synchronization and Gitignore Cleanup

- **Hyperparameter alignment:** Modified `chapter_4.tex` and `.cursor/rules/hyperparameters.mdc` to split weight decay and momentum, add learning rate decay ($\gamma = 0.99$), and correct weight decay to $\lambda = 10^{-4}$ (0.0001).
- **Learning rate decay:** Implemented `_get_decayed_lr` in `GenericClient` in `client.py` and integrated it across standard, FedMD, and FedKD training loops to apply exponential per-round decay.
- **Default parameters corrected:** Updated parameter defaults for public dataset size from 500 to 200 in `partitioning.py` and `strategy.py` to match the manuscript default value.
- **Gitignore and untracked logs:** Added `wandb/` and local logs `experiment_log.csv` and `experiment_log.jsonl` to `.gitignore` in `fedmaq-experiments`, and ran a cached git remove to untrack them.
- **Rule alignment:** Updated `.cursor/rules/` (`baselines.mdc`, `datasets-simulation.mdc`, `evaluation-metrics.mdc`, `hyperparameters.mdc`) to match the streamlined baseline count (8), corrected Dirichlet alpha values (0.1, 1.0, 10.0), and added auxiliary metrics.

### 2026-07-01 — Full Manuscript Audit (Ch. 1--4) and Codebase Hardening

**Audit scope:** All four released chapters compared line-by-line against the experiment codebase.

- Identified and fixed 7 issues across config, source, and scripts; all 19 tests continue to pass post-changes.
- **Bug: heterogeneous compute speed** — `client_comp_speed` was always pinned to `comp_max`; changed to `rng.uniform(comp_min, comp_max)` per §2.2 ("variable CPU frequencies").
- **Bug: unseeded stochastic rounding** — `DAdaQuantCompressionHook.compress()` used the global `np.random.rand`, breaking reproducibility; replaced with an injectable `np.random.Generator`.
- **Documented 5 canonical seeds** `[0, 42, 123, 456, 789]` in `conf/config.yaml` with multirun sweep command; `run.py` now passes `np.random.default_rng(cfg.seed)` to the hook (§4.3 statistical controls).
- **Config separation (CI vs. production):** `conf/experiment/default.yaml` restored to `total_rounds: 100`; new `conf/experiment/ci.yaml` provides `total_rounds: 2` override (`+experiment=ci`).
- **Dead config removed:** `kd_weight: 0.5` stripped from `fedmaq.yaml`; replaced with `kd_epochs: 1`, `server_kd_lr`, `server_kd_momentum` as explicit, named KD parameters.
- **Control group config added:** `conf/heterogeneity/uniform_memory.yaml` (8192 MB fixed) for §4.1 ablation; wired into `TelemetryFedAvg.__init__` via `uniform_memory_mb` key.
- **`run_server_side_kd` gains `epochs` param:** distillation passes over proxy dataset now configurable; body correctly nested inside both epoch and batch loops.
- **Stale default fixed:** `num_public_samples` hardcoded default in `aggregate_fit` corrected from 500 → 200.
- **Safe OmegaConf device resolution:** `cfg.get("device", DEVICE)` → `OmegaConf.select(cfg, "device", default=None)` in `evaluate_fn`.
- **Potential manuscript error flagged:** Table 4.1 `Weight Decay / Momentum ρ = 0.99` conflates two hyperparameters with a numerically suspect value; recommend splitting into `Momentum ρ = 0.9` / `Weight Decay = 0.0`.
- **FedDistill and CFD confirmed as zero-implementation stubs** — on schedule for Sep--Oct 2026 per Gantt.

### 2026-07-01 — Revised FedMAQ methodology, auxiliary metrics, and local telemetry logging

- Simplified local client training for FedMAQ in `client.py` to strictly perform task-loss cross-entropy minimization ($L_{local} = CE(\hat{y}, y)$), removing student-teacher mutual learning and model checkpoint persistence on the client.
- Redesigned server-side ensemble distillation for FedMAQ in `strategy.py` to dynamically construct client teacher models in memory from uploaded parameter updates, completely bypassing disk checkpoint files.
- Configured a uniform client computation speed (`comp_max`) in `TelemetryFedAvg.__init__` to simulate uniform hardware across the federation.
- Integrated macro-averaged Precision, Recall, and F1-score evaluation metrics on the server in `run.py`'s global and client-averaging evaluation paths.
- Setup local telemetry logging to generate isolated `experiment_log.jsonl` and `experiment_log.csv` run artifacts within each Hydra execution directory using `HydraConfig`.
- Successfully validated the implementation with the complete test suite and end-to-end 2-round MNIST simulation runs for FedAvg and FedMAQ.

---

## 11. Handoff recommendation

When ending a session, the `agent-handoff` skill indicates whether you should hand off to a new agent session for clean context. Do not recommend handoff without updating Section 6 and the changelog.
