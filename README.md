# Dynamic Pricing with Reinforcement Learning (Retail Price Optimization)

This project develops a **data-driven dynamic pricing system** using **Deep Reinforcement Learning (DRL)** to optimize retail pricing decisions under demand uncertainty.

Unlike static pricing or forecast-based optimization, this approach models pricing as a **sequential decision-making problem (MDP)** and learns pricing policies that maximize **long-term profit** while balancing **inventory and demand dynamics**.

---

## Problem Statement

Retail pricing decisions impact not only immediate revenue but also:
- future demand
- inventory levels
- long-term profitability

Traditional approaches rely on explicit demand models, which are often inaccurate in practice.

This project applies **Reinforcement Learning (RL)** to:
- learn pricing strategies directly from data
- optimize long-term reward under uncertainty
- evaluate performance using **business KPIs**

---

## Dataset Overview

- **676 observations** (product–time level)
- Target: `qty` (units sold / demand)

### Feature groups:
- **Price signals:** `unit_price`, `lag_price`, `freight_price`
- **Competition:** `comp_1`, `comp_2`, `comp_3`
- **Time features:** month, year, weekday, holiday
- **Product & demand context:** product category, product score, customers, volume

---

## Methodology

### 1) Data-driven Demand Model
A supervised model is trained to simulate demand:
- Model: Ridge Regression
- Target: `qty` (log-transformed)
- Performance:
  - MAE: 8.84
  - RMSE: 13.26
  - R²: 0.21

Added **stochastic noise** to simulate realistic demand uncertainty.

---

### 2) MDP Formulation

The pricing problem is modeled as a **Markov Decision Process**:

- **State:** inventory, time, past price, competitor prices, product signals  
- **Action:** price decision  
  - Discrete: 20 price levels  
  - Continuous: price range via PPO  
- **Reward:**
  Reward = Revenue − Cost − Holding Cost − End Inventory Penalty

This creates trade-offs between:
- margin vs volume
- inventory vs sell-through

---

### 3) Reinforcement Learning Models

#### Discrete Pricing (DQN family)
- DQN
- Double DQN
- Dueling DQN
- Double + Dueling DQN (best discrete model)

#### Continuous Pricing
- **Proximal Policy Optimization (PPO)**

---

### 4) Evaluation

- Multi-seed experiments (robustness)
- Reported as **mean ± standard deviation**
- Evaluated using both:
- RL reward
- Business KPIs

---

## Results

### RL Performance (Test Reward)
- **PPO:** 6952 ± 1130 (best overall)
- **Double + Dueling DQN:** 6738 ± 1164 (best discrete)

### Business Insights

| Metric | PPO | Double+Dueling DQN |
|------|-----|--------------------|
| Avg Profit | higher | slightly lower |
| Avg Price | higher | lower |
| Sell-through | lower | higher |
| Inventory | higher | slightly lower |

### Key Insight
- **PPO prioritizes margin (higher price)**
- **DQN balances margin and volume**

Demonstrates real-world trade-offs in pricing strategy.

---

## Key Contributions

- Data-driven demand modeling for realistic RL environment
- Comparison of **discrete vs continuous pricing strategies**
- Robust evaluation using **multi-seed experiments**
- Integration of **business KPIs** (not just ML metrics)
- Demonstration of trade-offs between revenue and inventory

---

## Limitations

- Small dataset (676 observations)
- Demand model has low R² (high uncertainty)
- Reward function parameters are manually designed
- No real-world deployment validation

---

## Future Work

- Tune reward weights (inventory vs revenue)
- Add service-level constraints (target sell-through)
- Extend to multi-product pricing (cross-elasticity)
- Incorporate promotions and seasonality dynamics
