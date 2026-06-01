# Thesis Metadata & Domain Context

This document maintains the high-level research context of the Master of Science in Computer Science (MS CS) thesis. Future AI agents and tools will refer to this file to align simulations, optimization models, algorithms, and experimental configurations with your academic work.

---

## 1. Administrative Details

- **Thesis Title**: Algorithmic Codesign of Adaptive and Lightweight Federated Distillation for Healthcare IoT
- **Candidate**: Christian Joseph Bunyi
- **Degree**: Master of Science in Computer Science
- **Advisor**: Fritz Flores
- **Committee Members**:
  - [Committee Member 1]
  - [Committee Member 2]
- **Laboratory**: Dr. Andrew L. Tan Data Science Institute
- **Institution**: De La Salle University
- **Target Defense Date**: [Insert Target Date]

---

## 2. Research Problem & Questions

### Problem Statement

Integrating compression techniques into Federated Distillation (FD) reduces communication and computation overhead, but introduces a severe "Compression-Utility Trade-off" in non-IID (heterogeneous) data settings. Current static baseline methods (e.g., Compressed Federated Distillation - CFD) apply fixed quantization, causing significant information loss and degrading model performance on complex or high-dimensional feature spaces.

Conversely, state-of-the-art dynamic compression algorithms (like DynFed) base their adaptive triggers heavily on hardware resources. In smart healthcare / medical wearables, this creates a critical blind spot: if a wearable device has low memory but captures highly complex, skewed data (e.g., sudden erratic movements or a fall), resource-aware algorithms will blindly over-compress the updates, destroying life-critical motion features. The proposed solution is a hybrid "data- and resource-aware" adaptive quantization framework, which uses hardware constraints to bound the maximum bit-width and client-side data complexity to optimize the final compression rate.

### Core Research Questions (RQs)

1. **RQ1**: How can we quantify the non-IID complexity of selected Healthcare IoT datasets to inform the data-aware aspect of the adaptive compression mechanism?
2. **RQ2**: How can we design a hybrid dynamic quantization algorithm that minimizes communication and computational overhead by bounding maximum compression bit-widths to device resource constraints, while preserving activity recognition accuracy by optimizing the final bit-width according to the measured data heterogeneity?

---

## 3. Core Contributions & Novelty

- **Contribution 1**: Quantifying the non-IID complexity of selected Healthcare IoT datasets to establish client data complexity profiles.
- **Contribution 2**: A hybrid data- and resource-aware adaptive quantization framework that bounds compression to device memory and optimizes based on statistical skew.
- **Contribution 3**: An open-source, reproducible simulation framework on the Flower FL platform utilizing Human Activity Recognition (HAR) wearable benchmarks.

---

## 4. Key Terminology & Domain Vocabulary

- **FL**: Federated Learning
- **FD**: Federated Distillation
- **IoMT**: Internet-of-Medical-Things
- **HAR**: Human Activity Recognition (predicting physical routines, gaits, or falls from mobile sensors)
- **FedAvg / FedProx**: Classic baseline federated aggregation optimization models
- **Flower**: Core open-source FL simulation framework selected for implementation
- **UCI Smartphone / MobiAct**: Standard gold-standard wearable datasets used for testing

---

## 5. Standard Mathematical Notation

- **Data Samples**: $n$ total samples, $n_k$ samples on client $k$.
- **Model Weights**: Global weights at step $t$ are $w_t$.
- **Learning Rate**: $\eta$ (eta).
- **Gradient Update**: $g_k$ for client $k$.
- **Quantization Bit-Width**: $b_k$ for client $k$ (the target optimization parameter).
- **Objective Function**: $f(w)$ or $F_k(w)$ for local objectives.
