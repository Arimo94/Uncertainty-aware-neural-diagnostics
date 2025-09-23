# Probabilistic Machine Learning for Uncertainty-Aware Diagnosis of Industrial Systems

<!-- [![Paper](https://img.shields.io/badge/paper-PDF-blue)](./root.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)](https://www.python.org/) -->

This repository contains the implementation of our diagnostic framework that integrates **ensemble probabilistic machine learning** into **consistency-based fault diagnosis**. The method addresses a key challenge in data-driven diagnostics: quantifying predictive uncertainty to reduce false alarms while maintaining robust fault detection and isolation.

---

## 🔍 Overview

Traditional consistency-based diagnosis relies on physical models to generate residuals that trigger alarms when system behavior deviates from expectations. While effective, these models are often expensive to develop.  

Our approach replaces physical residuals with **data-driven residuals based on deep neural networks**, and augments them with **uncertainty quantification**:

- **Aleatoric uncertainty**: captures inherent system noise  
- **Epistemic uncertainty**: captures model uncertainty due to unseen conditions  

By explicitly modeling both, our framework reduces false alarms and improves reliability in real-world diagnostic settings.

---

## ⚙️ Framework

<!-- <p align="center">
  <img src="docs/fig_framework.png" alt="Framework Diagram" width="500"/>
</p> -->

The pipeline consists of:

1. **Residual generation** from structural analysis of system equations  
2. **Ensemble probabilistic neural networks (PNNs)** to model system dynamics  
3. **Uncertainty quantification** to distinguish between noise and out-of-distribution inputs  
4. **Consistency-based decision logic** with adaptive thresholds  

---

## 📊 Case Studies

We evaluate the framework on three diagnostic benchmarks:

- 🧪 **Two-tank system (simulation)**: actuator, sensor, leakage, and obstruction faults  
- 🚛 **Aftertreatment in heavy trucks (experimental)**: clogging faults in urea dosing system  
- 🚗 **Gasoline engine air-path (industrial benchmark)**: sensor and leakage faults in a turbocharged engine  

The framework consistently reduced false alarms while preserving sensitivity to true faults across all domains.

---

## 📂 Repository Structure