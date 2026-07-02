---
description: synchronizes experiment code & config variables to LaTeX thesis definitions
---

1. Check the LaTeX manuscript files in [fedmaq-manuscript/](../fedmaq-manuscript/) (e.g., `chapters/` or main LaTeX files) to identify changes in hyperparameters, formulas, or system definitions.
2. Read the canonical rules in [.cursor/rules/manuscript-alignment.mdc](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/.cursor/rules/manuscript-alignment.mdc) to check active constraints.
3. Validate and synchronize parameters in [default.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/experiment/default.yaml) and [preliminary.yaml](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/conf/experiment/preliminary.yaml) (such as client counts $K$, batch sizes $B$, epochs $E$, learning rate decay $\gamma$, or proxy size $\|D_{pub}\|$).
4. Verify that the client delay simulation in [client.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/client.py) and server distillation delay in [strategy.py](file:///c:/Users/Quirora/Documents/GitHub/fedmaq-experiments/src/fedmaq/core/strategy.py) match the mathematical formulations defined in the manuscript.
5. Check that precision-scaling allocations inside the strategy class match the soft quality targets (Formulations 0 to 4).
6. Run `/test` to verify that all mathematical formulations and telemetries pass their test assertions.
