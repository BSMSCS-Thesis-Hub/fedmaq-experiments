---
description: scaffolds a new FL baseline implementation and updates registry
---

To implement a new SOTA baseline:

1. Create a Hydra configuration file under `conf/algorithm/{name}.yaml`.
2. Implement strategy logic, client training loops, or algorithm-specific components inside [strategy.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy.py) and [client.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/client.py). Use [baselines/](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/baselines/) for standalone compression/quantization helper modules.
3. Integrate the baseline strategy selection into the main training runner in [run.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/scripts/run.py).
4. Update [.cursor/project/baseline_registry.md](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/.cursor/project/baseline_registry.md) with the new algorithm details, setting status to `[In Progress]` or `[Complete]`.
5. Run the verification tests using the `/test` command to ensure correct initialization.
