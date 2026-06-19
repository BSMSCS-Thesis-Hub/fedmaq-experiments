---
name: scaffold-baseline
description: Add a SOTA baseline under src/fedmaq/baselines/ with Hydra config and registry entry
---

# Scaffold Baseline

When asked to add a baseline (e.g. FedProx, DynFed):

1. Read `.cursor/rules/baselines.mdc` and `.cursor/project/baseline_registry.md`.
2. Create package under `src/fedmaq/baselines/<name>/` (client, server, strategy as needed).
3. Add `conf/algorithm/<name>.yaml`.
4. Add a row to `baseline_registry.md` with config path and `[In Progress]` status.
5. Follow `flower-patterns.mdc` and `evaluation-metrics.mdc` for WandB logging.
