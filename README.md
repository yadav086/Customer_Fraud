# Customer Fraud Detection Pipeline

An end-to-end, modular machine learning and deep learning pipeline optimized to detect fraudulent transactions in highly imbalanced datasets. This architecture handles data cleaning, automatic feature selection via RFECV, SMOTE oversampling, and includes an Artificial Neural Network wrapped to fit seamlessly alongside standard machine learning steps.

---

## 📂 Project Directory Structure

```text
Customerfraud/
├── .venv/                      # Managed Python 3.12 virtual environment
├── artifacts/                  # Trained model targets and visualization plots
├── data/
│   └── credit_card_fraud_2026.csv  # Raw transaction data source
├── scr/                        # Pipeline source code module folder
│   ├── __pycache__/
│   ├── best_fraud_model.keras  # Saved TensorFlow checkpoint binary
│   ├── config.yaml             # Global pipeline configurations
│   ├── customer_fraud.ipynb    # Jupyer development scratchpad notebook
│   ├── load_ann_main.py        # Independent Neural Network pipeline runner
│   ├── load_ANN_model.py       # Standalone Deep Learning structural graph
│   ├── load_config.py          # Configuration parser utility
│   ├── load_csv.py             # Data loading and collection module
│   ├── load_features.py        # Feature/Target partitioning module
│   ├── load_main.py            # Local pipeline test script
│   ├── load_metrics.py         # Advanced metrics evaluator (PR-AUC, ROC-AUC)
│   ├── load_model.py           # Machine learning pipeline assembly (RFECV, LightGBM)
│   ├── load_preprocess.py      # Data preprocessing encoders
│   ├── load_remove_outlier.py  # Outlier cleaning algorithms
│   └── load_spilt_data.py      # Stratified data partitioning script
├── .python-version             # Python version pin for uv (3.12)
├── main.py                     # Primary pipeline execution workflow entrypoint
├── pyproject.toml              # Unified package declaration file
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Frozen dependency manifest
└── uv.lock                     # UV cryptographic environment state lockfile
```

---

## 🛠️ Tech Stack & Dependencies

- **Runtime & Management:** Python 3.12, [uv](https://github.com) (Package resolver)
- **Core Processing:** NumPy, Pandas, Scikit-Learn
- **Imbalance Handling:** Imbalanced-Learn (`SMOTE`)
- **Gradient Boosting:** LightGBM
- **Deep Learning Subsystem:** TensorFlow 2.16+, SciKeras (`KerasClassifier`)

---

## 🚀 Environment Setup & Installation

This project uses `uv` for ultra-fast, clean environment synchronization.

### 1. Build Environment Fresh
To fix potential Windows binary resource locks or cross-drive performance degradations, force-create a fresh standalone workspace environment from your terminal:
```bash
# Force wipe old environments if present, then build fresh with pip seeded
uv venv --python 3.12 --seed --force
```

### 2. Install Project Dependencies
Run the package sync installer in explicit copy mode to copy fresh distribution wheel targets directly into your project:
```bash
uv pip install -r requirements.txt --link-mode=copy
```

---

## 🧠 Strategic Pipeline Architecture

1. **Path-Resolution Safety:** The main script `main.py` sits at the root, while logic files are nested inside `scr/`. The entrypoint injects `scr/` dynamically into `sys.path` to guarantee robust module lookups.
2. **Stratified Splitting:** Fraud represents a rare minority class. Data partitioning uses `StratifiedShuffleSplit` logic to ensure both training and validation sets perfectly mirror authentic target prevalence.
3. **Information Leakage Prevention:** Sampling techniques (`SMOTE`) are nested **inside** an `imblearn.pipeline.Pipeline` rather than standard scikit-learn pipelines. This guarantees synthetic rows are calculated **exclusively for training data during `.fit()`** and never leak into out-of-sample data.
4. **Windows Multiprocessing Protection:** High-performance functions utilizing multiple cores (`n_jobs=-1` inside LightGBM or RFECV) are strictly executed under the `if __name__ == '__main__':` paradigm to prevent infinite subprocess fork-bomb crashes on Windows systems.

---

## 🏃 Running the Pipeline

To execute the data preprocessing, resampling, feature engineering, and model training workflow seamlessly, run the root entry point:

```bash
python main.py
```

### Production Log Expectation
```text
2026-09-19 11:15:01,234 - INFO - Initializing fraud detection pipeline execution...
2026-09-19 11:15:02,512 - INFO - Step 1 Complete: Successfully loaded source CSV. Raw shape: (284807, 31)
2026-09-19 11:15:03,119 - INFO - Step 2 Complete: Features extracted. Numeric columns: (284807, 29), Categorical columns: (284807, 1)
2026-09-19 11:15:03,400 - INFO - Step 3 Complete: Preprocessing configurations successfully created.
2026-09-19 11:15:03,450 - INFO - Step 4 Complete: Model pipeline setup defined.
2026-09-19 11:15:04,810 - INFO - Step 5 Complete: Data successfully split into training and test chunks.
2026-09-19 11:15:04,811 - INFO -  -> Training Data Size: (227845, 30) | Test Data Size: (56962, 30)
2026-09-19 11:15:04,812 - INFO -  -> Fraud Prevalence in Training Set: 0.1730%
2026-09-19 11:15:04,815 - INFO - Step 6: Launching model pipeline training (Preprocess -> Resample -> Train)...
...
2026-09-19 11:16:45,992 - INFO - Pipeline pipeline training completed successfully!
2026-09-19 11:16:46,010 - INFO - Starting model evaluation on test data...
```

---

## 📊 Evaluation Focus

Standard accuracy flags are omitted since they obscure model performance on highly skewed populations. The operational performance metrics tracked are:
- **Precision-Recall AUC (PR-AUC):** Tracks real capture efficiency against operational workflow friction (False Alarms).
- **Recall / Sensitivity:** Focuses on minimizing costly False Negatives (Missed fraud attacks).
- **Confusion Matrix Breakdown:** Real-time counts logging true negatives, false alarms, missed alerts, and caught items.
