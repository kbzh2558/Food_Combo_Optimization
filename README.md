# 🍱📈 A Data-Driven Approach for Decision Making in Food Combo Offer

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=Python)
![Optimization](https://img.shields.io/badge/Linear_Programming-Gurobi-green)
![Demand Model](https://img.shields.io/badge/Forecasting-NeuralProphet-orange)
![Retail Analytics](https://img.shields.io/badge/retail-food_combo_analysis-purple)

> **Authors**  
> Kaibo Zhang, Mingshu Liu, Yanfei Wu, David Min, and Charlie Tomlinson  
> **Affiliation**: McGill University (MGSC 407 - Retail Analytics)  
---

## 🧠 Overview

This project presents a full-stack data-driven framework to optimize **multi-item food combos** for Couche-Tard, a leading convenience store chain in Quebec. We integrated transaction data analysis, demand forecasting, optimization modeling, and survey-based validation to recommend **relevant, profitable, and nutritionally coherent bundles**.

**Key contributions:**
- 📊 Extracted frequent co-purchase patterns using **Association Rule Mining**.
- 🔮 Modeled demand with **price memory, cross-price elasticity**, and **seasonality** using NeuralProphet.
- 🔧 Implemented a **Linear Programming** model to optimize combos under real-world constraints.
- 🧪 Conducted external validation via survey with **Net Promoter Score (NPS)** analysis and NLP comment mining.
- 📦 Proposed final combo structures balancing **consumer affinity and margin**.

---

## 🛒 Dataset Description

### 🔹 Internal Data
- **Source**: Couche-Tard POS system (Jan 2021 – Feb 2025)
- **Details**: Over 5,000 items, store/site identifiers, price, quantity sold, transaction timestamp
- **Metadata**: Brand type (private/national), procurement cost, internal co-purchase affinity

### 🔹 External Data
- **Consumer Survey**: 404 valid Canadian responses via Prolific  
- **Task**: Simulated shopping + combo rating (0–10)  
- **Goal**: Validate proposed bundles using stated preferences

---

## 🧪 Methodology

### 1. Association Rule Mining (Apriori)
- Reduced the item space to top-performing food & drink SKUs
- Identified co-purchase patterns to guide combo design

### 2. Predictive Demand Modeling
- Regression + NeuralProphet-based hybrid model:
  - 🔁 Lagged price (past 3 days)
  - 🔗 Cross-product price effects (correlation ≥ 0.6)
  - 📆 Seasonal/weekly trends
- RMSE ≈ 247 units; strong performance on trend capture

### 3. Linear Programming Optimization
- Objective: maximize α·profit + (1–α)·bundle relevance  
  - α = 0.6 (margin prioritized)
- Constraints:
  - 🥗 Nutrition (1 food, ≤2 beverages)
  - 🏷️ No same-category duplication
  - 🏭 Supplier conflict exclusion
  - 💵 At least one discounted item per combo

### 4. Price Elasticity Analysis
- Log-log regressions across categories (private vs. national brands)
- Highest sensitivity: **energy drinks, salty snacks**
- Price-insensitive: **candy, soft drinks (private label)**

### 5. External Survey & NPS Validation
- NPS computed from ratings (Promoters 9–10, Detractors 0–6)
- Best NPS combo: 🧀 Fruit & Cheese + 🥪 Steak Panini (NPS = –27.5)
- Worst NPS combo: 🧁 Muffin + 🌭 Hot Dog + ☕ Coffee + 🥤 Red Bull (NPS = –77)
- Key insight: **less is more** — smaller, familiar combos outperform overstuffed ones

---

## 📈 Results Summary

| Combo                               | Items                          | Price   | NPS     |
|-------------------------------------|--------------------------------|---------|---------|
| Balanced Combo (Best)              | Fruit & Cheese + Panini        | \$11.59 | –27.47  |
| Control Combo (Legacy)            | Baguette, Lay’s, Pepsi, Eska   | \$12.75 | –32.43  |
| Overloaded Combo (Worst)          | Muffin, Hot Dog, Red Bull, Coffee | \$10.52 | –77.00  |

---

## 💡 Insights & Recommendations

- **Simple Combos Win**: 2–item combos maximize appeal and align with grab-and-go behavior.
- **Avoid Redundancy**: Multiple drinks or overstuffed bundles confuse and deter consumers.
- **Target Price Zone**: \$5–\$15 hits the sweet spot for willingness to pay.
- **Cater to Vegetarians**: Plant-forward combos scored better among non-meat consumers.
- **Strategic Discount Framing**: Highlight the discounted drink to boost perceived value.

---

## 🔄 Future Work

- A/B test optimized combos in select stores to validate impact
- Develop seasonal or regional combo variants
- Integrate loyalty card data to personalize recommendations
- Use NLP (e.g., BERT) for deeper comment sentiment analysis

---

## 📌 References
- [1] Cohen & Perakis. "Promotion Optimization in Retail", SSRN 2018  
- [2] Triebe et al. "NeuralProphet: Explainable ForecastiDng at Scale", arXiv 2021  

---

## 📬 

---
