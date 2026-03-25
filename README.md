# Autonomous Intelligent Irrigation System
> Deep Reinforcement Learning for Water-Efficient Agriculture in Desert Regions

---

## Overview

This project develops a smart, autonomous, and water-efficient irrigation system designed for **arid and desert environments** specifically the Saharan region of El Oued, Algeria. By combining **LSTM-based predictive models** with a **Deep Q-Network (DQN) reinforcement learning agent**, the system makes real-time irrigation decisions tailored to crop needs and local climate conditions, targeting tomato cultivation.

---

## Motivation

Agricultural water management in desert climates is a critical challenge. Conventional irrigation wastes water, reduces yields, and is unsustainable at scale. This project addresses that gap by:

- Minimizing water waste in high water-stress zones
- Maximizing crop yields through data-driven, personalized irrigation schedules
- Reducing operational costs via intelligent resource management
- Demonstrating the viability of AI-driven precision agriculture in harsh climates

---

## System Architecture

The pipeline consists of three interconnected components:

### 1. LSTM Model — Soil Water Content Prediction
- **Inputs:** Climatic variables + irrigation history (see `docs/data.md`)
- **Temporal window:** 4–7 days
- **Output:** Predicted soil water content (SWTD) for the next day

### 2. LSTM Model — Yield Estimation
- **Inputs:** Climate data, historical irrigation, full-season SWTD
- **Output:** End-of-season biomass/yield estimate (proxied via DSSAT's `CWAD` variable)

### 3. DQN Agent — Irrigation Decision Optimizer
- **Action space:** 12 discrete irrigation levels (0–60 mm/day)
- **Reward function:** Economic yield − water cost
- **Goal:** Learn an optimal policy that maximizes profit while conserving water

---

## Tech Stack

| Component | Technology |
|---|---|
| Predictive modeling | LSTM (Deep Learning) |
| Decision-making | Deep Q-Network (DQN) |
| Agronomic simulation | DSSAT / CROPGRO-Tomato (via DSSATTools) |
| Climate data | NASA POWER API |
| Preprocessing | Z-Score normalization, PCA |
| Evaluation metrics | RMSE, R² |

---

## Regional Context

- **Location:** Wilaya d'El Oued, Algeria : extreme Saharan climate
- **Target crop:** Tomato (heat-adapted local varieties)
- **Irrigation infrastructure:** Automated mobile pivot systems

---

## Project Report

The complete methodology  including data collection, DSSAT simulation setup, feature engineering, model training procedures, and result figures  is documented in French in:

```
Rapport/Système_d_Irrigation_Intelligent.pdf
```

---

## Acknowledgements

This work draws inspiration from deep reinforcement learning approaches to irrigation optimization, particularly *"Optimizing Irrigation Efficiency using Deep Reinforcement Learning in the Field"* (Ding & Du). The implementation is adapted to the El Oued tomato growing context using NASA POWER climate data and DSSAT-based agronomic simulations.

---

## License

MIT License  see [`LICENSE`](./LICENSE).