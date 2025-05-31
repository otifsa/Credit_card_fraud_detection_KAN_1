# Dynamic Oversampling‑Driven Kolmogorov–Arnold Networks (KAN)  
**Credit‑Card Fraud‑Detection Framework**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Full, reproducible pipeline that couples **Kolmogorov–Arnold Networks (KAN)** with **dynamic SMOTE / GAN oversampling** and an **ensemble feature‑selection** strategy to achieve state‑of‑the‑art credit‑card‑fraud detection on three public benchmarks.  
> The methodology and results are detailed in:  
> *AKOUHAR M. et al., “Dynamic Oversampling‑Driven Kolmogorov–Arnold Networks for Credit-Card Fraud Detection: An Ensemble Approach to Robust Financial Security,” 2025.*

---

## Key Features

| Module | Highlights |
|--------|------------|
| Pre‑processing | Mean imputation, categorical encoding, Min–Max scaling |
| Feature selection | Five meta‑heuristics (GA, PSO, ACO, Aquila Optimizer, Grey Wolf Optimizer + WOA). Final subset = features chosen by ≥ 3 optimisers |
| Baselines | DNN, CNN, LSTM, GRU, KAN **without** oversampling |
| Dynamic SMOTE | Minority‑class ratio adapted automatically at each training epoch |
| Dynamic GAN | Fully connected GAN that generates synthetic minority samples on the fly |
| Explainability | End‑to‑end SHAP scripts for every KAN variant |
| Statistics | Wilcoxon signed‑rank test confirms KAN + GAN superiority (*p* = 0.00195) |
| Efficiency | End‑to‑end inference under seven seconds for 590 k IEEE‑CIS transactions |

---

## Repository Structure

```
Credit_card_KAN_code/
├── Datasets/                      # Raw public datasets
│   ├── creditcard.csv             # European dataset
│   ├── Sparkov_data.parquet       # Sparkov synthetic dataset
│   └── IEEE-fraud-detection.parquet
├── Preprocessing/                 # Cleaning & encoding scripts
├── Features selection/
│   └── FS_dataset{1,2,3}/
│       └── {GA,PSO,ACO,AO,GWO_WOA}.py
├── Experiment 1/                  # Baselines (no oversampling)
│   └── Exp1_{1,2,3}/
├── Experiment 2 SMOTE/            # Dynamic‑SMOTE experiments
│   └── Exp2_smote_{1,2,3}/
├── Experiment 2 GAN/              # Dynamic‑GAN experiments (+ SHAP)
│   └── Exp2_gan_{1,2,3}/
└── README.md  (you are here)
```

---

## Installation

1. **Clone** the repository  
   ```bash
   git clone https://github.com/<your‑handle>/credit‑card‑KAN.git
   cd credit_card_KAN
   ```
2. **Create** and activate a fresh Python 3.10 environment (conda or venv).  
3. **Install** dependencies  
   ```bash
   pip install -r requirements.txt
   ```
4. **Download** the three public datasets into the `Datasets/` directory.

---

## Reproducing the Experiments

1. **Pre-process** each dataset using the scripts in `Preprocessing/`.  
2. **Run** all five feature‑selection scripts for the target dataset; use the provided helper to merge them into the consensus feature list.  

### Experiment 1 – Baselines (no oversampling)  
Execute the training script for each model located in `Experiment 1/Exp1_*`.

### Experiment 2a – Dynamic SMOTE  
Launch the SMOTE variants inside `Experiment 2 SMOTE/Exp2_smote_*`.

### Experiment 2b – Dynamic GAN  
Execute the GAN variants located in `Experiment 2 GAN/Exp2_gan_*`.  
Optional SHAP scripts generate feature‑importance visualisations.

All runs export:
* `metrics.json` – accuracy, precision, recall, F1, AUC, MCC  
* `model_best.*` – saved model weights  
* `shap_summary.png` (GAN variants)

-

## License

Distributed under the **MIT License**. See the `LICENSE` file for full text.

---

## Acknowledgements

* **Datasets**: European Credit‑Card Fraud (Kaggle), IEEE-CIS Fraud Detection (Kaggle), Sparkov Generator  
* **Meta-heuristics**: Implementations adapted from the *MEALPY* library  
* **GAN training loop**: Based on official Keras examples  

For questions or collaboration proposals, please open an issue or contact **m.akouhar@gmail.com**.
