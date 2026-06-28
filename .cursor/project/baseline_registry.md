# Baseline Registry

Maps each SOTA target to implementation status. Update when porting into `src/fedmaq/baselines/`.

| Algorithm  | Group             | Paper / Citation        | Config                           | Status        | Notes |
| ---------- | ----------------- | ----------------------- | -------------------------------- | ------------- | ----- |
| FedAvg     | Seminal Controls  | McMahan et al., 2017    | `conf/algorithm/fedavg.yaml`     | [Not Started] |       |
| FedProx    | Seminal Controls  | Li et al., 2020         | `conf/algorithm/fedprox.yaml`    | [Not Started] |       |
| FedPAQ     | Pure Quantization | Reisizadeh et al., 2020 | `conf/algorithm/fedpaq.yaml`     | [Not Started] |       |
| DAdaQuant  | Pure Quantization | Hönig et al., 2022      | `conf/algorithm/dadaquant.yaml`  | [Not Started] |       |
| FedMD      | Pure KD           | Li et al., 2019         | `conf/algorithm/fedmd.yaml`      | [Not Started] |       |
| FedDistill | Pure KD           | Song et al., 2024       | `conf/algorithm/feddistill.yaml` | [Not Started] |       |
| FedKD      | Hybrid Q+KD       | Wu et al., 2022         | `conf/algorithm/fedkd.yaml`      | [Not Started] |       |
| CFD        | Hybrid Q+KD       | Sattler et al., 2022    | `conf/algorithm/cfd.yaml`        | [Not Started] |       |
| FedMAQ     | Proposed SOTA     | Bunyi et al., 2026      | `conf/algorithm/fedmaq.yaml`     | [Not Started] |       |

**Status:** `[Not Started]` | `[In Progress]` | `[Complete]`
