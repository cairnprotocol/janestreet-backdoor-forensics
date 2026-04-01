# LLM Forensic Evaluation Framework (Jane Street Investigation)

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

We identify a pathway-specific failure: numerical reasoning is non-invariant to surface form perturbations, while decision-level outputs remain partially invariant under the same transformations.

---
# Key Findings

1. **Surface-form sensitivity**
   Small syntactic changes (e.g., punctuation, delimiter wrapping, encoding) produce measurable shifts in early-layer activations and downstream numeric reasoning.

2. **Computation–decision dissociation**
   Models can compute expected value incorrectly while maintaining consistent decision direction across equivalent formulations.

3. **Format-dependent failure modes**
   JSON formatting induces systematic failure (malformed or non-strict outputs), while XML remains stable across models.

4. **Cross-model consistency**
   These patterns replicate across multiple dormant models under aligned conditions, suggesting structural rather than stochastic effects.

5. **No single-token trigger identified**
   Evidence suggests compound or pathway-based triggers rather than discrete backdoor strings, based on systematic negative results across trigger sweeps.

These findings were derived from controlled perturbation experiments, not post-hoc interpretation.

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

# Development Trace (Condensed)

The investigation evolved in stages:

1. Initial probing focused on single-turn formatting perturbations.
2. Early experiments did not reveal clear anomalies under simple perturbations, motivating expansion into compositional and multi-turn probes.
3. Anomalies were clustered and rerun using trigger ladders.
4. Cross-model comparisons confirmed consistency of failure patterns.
5. Targeted diagnostics were developed for Model 3 after anomalous behavior.

The pipeline was not predefined; it was constructed in response to observed failures.

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

Failures are evaluated along the following dimensions:

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

# What This Rules Out

This investigation provides evidence against:

- single-token backdoor triggers
- purely stochastic output variability
- format-agnostic reasoning pipelines
- decision-level randomness

Instead, failures appear structured and pathway-dependent.

---

# Notes

* Findings are **empirical and evidence-based**, not causal claims
* Some behaviors (especially Model 3) remain **under active investigation**
* Activation analyses are **correlational, not causal**

---

# Contribution

This work demonstrates:

> This work demonstrates that LLM failures can be systematically elicited and analyzed through controlled perturbations, revealing structured, pathway-dependent behavior rather than purely stochastic variation.

---

# Contact

For questions or collaboration, feel free to reach out.
