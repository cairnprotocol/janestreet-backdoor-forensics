# Janestreet Backdoor Forensics
This repository contains my approach to identifying backdoor triggers trained into 3 different language models. Includes evaluation scripts detailing the exploration, detection methodology, and final analysis.

# LLM Forensic Evaluation Framework

### Detecting Invariant Violations and Decision–Computation Dissociation

This repository contains a **forensic evaluation system** designed to analyze latent failure modes in large language models, including:

* syntactic decision boundary violations
* format-sensitive failures (JSON/XML)
* computation–decision divergence
* cross-model behavioral inconsistencies
* compositional and multi-turn vulnerabilities

---

# Overview

This project investigates a central question:

> Are model outputs invariant to semantically equivalent inputs?

We identify and analyze **pathway-specific failures**, where:

* numerical reasoning is **non-invariant to surface form**
* decision outputs can **diverge from correct computation**

---

# System Architecture

The evaluation pipeline is modular and fully reproducible:

```
Condition Generation
   ↓
Execution (API / jsinfer)
   ↓
Response Scoring
   ↓
Hotspot Ranking
   ↓
Trigger Ladder Reruns
   ↓
Cross-Model Comparison
   ↓
Targeted Diagnostics
```

---

# Repository Structure

## Core Harness

* `behavioral_anomaly_harness.py`
  → defines probes, triggers, scoring, anomaly detection

* `multi_turn_sensitiveescaltion_913.py`
  → multi-turn trajectories (injection, drift, escalation)

* `MoE_test.py`
  → compositional + cross-language + style ablation experiments

---

## Experimental Pipeline

* `generate_experiment.py`
  → condition generation

* `run_experiment.py`
  → execution layer

* `score_outputs.py`
  → scoring + classification

* `rank_hotspots.py`
  → anomaly clustering

* `rerun_candidates.py`
  → trigger ladder refinement

* `make_report.py`
  → reporting

---

## Cross-Model Analysis

* `compare_models.py`
  → condition-aligned comparison across models

Tracks:

* semantic correctness
* decision direction
* EV computation
* generation mode

---

## Model 3 Diagnostics

* `probe_model3.py`
  → minimal liveness battery

* `model3_diagnostic_runner.py`
  → structured diagnostic suite:

  * EV tasks
  * invariance tests
  * JSON vs freeform
  * execution error classification

---

## Validation

* `smoke_test.py`
  → baseline sanity checks

---

# Failure Classification

We classify failures across multiple dimensions:

| Dimension            | Description                             |
| -------------------- | --------------------------------------- |
| Semantic correctness | correct vs structurally valid but wrong |
| Decision direction   | take vs decline vs inconsistent         |
| EV correctness       | numeric accuracy vs drift               |
| Generation mode      | normal / terse / malformed / empty      |
| Format adherence     | JSON/XML correctness                    |

---

# Key Experimental Axes

* **Representation**: canonical vs synonym vs reordered
* **Structure**: JSON, XML, freeform
* **Position**: injection at different turns
* **Trajectory**: multi-turn drift and escalation
* **Language**: English, French, Spanish
* **Style**: poetry vs prose

---

# Model Comparison

All comparisons are performed on **aligned conditions**:

> Differences are attributed to model behavior, not prompt variation.

Comparison dimensions:

* semantic differences
* decision flips
* EV deviations
* generation inconsistencies

---

# Execution Error Handling

We explicitly separate:

* model failures
* infrastructure failures

Tracked:

* timeout
* rate limits
* transport errors
* malformed outputs
* off-task behavior

---

# What This Enables

This system provides:

* reproducible forensic evaluation
* structured anomaly detection
* cross-model behavioral comparison
* systematic probe generation
* iterative trigger refinement

---

# Notes

* Findings are **empirical and evidence-based**, not causal claims
* Some behaviors (especially Model 3) remain **under active investigation**
* Activation analyses are **correlational, not causal**

---

# Contribution

This work demonstrates:

> A modular forensic framework for detecting invariant violations, structural failures, and cross-model divergences in LLM behavior.

---

# Contact

For questions or collaboration, feel free to reach out.
