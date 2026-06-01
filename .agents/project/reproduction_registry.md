# Baseline Reproduction Registry

This registry tracks the organization of baseline algorithm reproductions inside `reproductions/`. Any AI agent or author editing, optimizing, or running simulations should consult and update this registry.

---

## 1. Active Baseline Reproductions

| Algorithm Name   | Project Directory                                                                                                             | Simulation API                                  | Datasets       | Status       | Notes / Progress                                                                              |
| :--------------- | :---------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------- | :------------- | :----------- | :-------------------------------------------------------------------------------------------- |
| **FedAvg MNIST** | [reproductions/fedavg_mnist/](file:///c:/Users/Quirora/Documents/GitHub/reproducibility-artifacts/reproductions/fedavg_mnist) | Classic Simulation API (`start_simulation`)     | MNIST          | `[Complete]` | Self-contained project with flat imports. Runs with `uv run python main.py`. Pytests passing. |
| **FedProx**      | [reproductions/fedprox/](file:///c:/Users/Quirora/Documents/GitHub/reproducibility-artifacts/reproductions/fedprox)           | Modern Flower App API (`ClientApp`/`ServerApp`) | MNIST, FEMNIST | `[Complete]` | Self-contained baseline using newest Flower framework style. Synchronized with `uv sync`.     |

---

## 2. Dynamic Metric Tracking & Logs

- FedAvg central metrics curve visual is cached in [reproductions/fedavg_mnist/docs/centralized_metrics.png](file:///c:/Users/Quirora/Documents/GitHub/reproducibility-artifacts/reproductions/fedavg_mnist/docs/centralized_metrics.png).
- Experimental run configurations and metric plots are automatically outputted to `reproductions/<baseline>/docs/results/` during runtime (and excluded from Git tracking via `.gitignore`).

---

## Progress Definitions

- **`[Not Started]`**: Empty directory or raw placeholders.
- **`[Draft]`**: Baseline files copied, but dependencies are not fully configured or imports are broken.
- **`[Complete]`**: Fully self-contained `pyproject.toml` established, `uv sync` compiles cleanly, verification tests pass, and dry runs execute successfully.
