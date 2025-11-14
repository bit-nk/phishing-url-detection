# Phishing URL Detection — Reproducible Machine Learning Pipeline

This repository implements a fully reproducible phishing-URL detection pipeline using lexical feature extraction, structured dataset merging, and machine-learning models. It follows the Data Ethics & Reproducibility Workshop guidelines and includes complete documentation, datasets, and reproducible scripts.

---

## 📁 Repository Structure

```
phishing-url-detection/
│
├── src/                     # Core pipeline scripts
│   ├── build_phishing_dataset.py
│   ├── evaluate_models.py
│
├── project_data/            # Raw datasets (CSV/ARFF files placed here)
│   └── sample_data.csv
│
├── out/                     # Auto-generated outputs (ignored by git)
│
├── reports/
│   ├── figures/             # Final visualizations for publication
│   └── results/             # Final evaluation tables
│
├── metadata/
│   ├── DATA_README.md
│   ├── ETHICS.md
│   └── LICENSE
│
├── notebooks/               # Optional Jupyter notebooks
│
├── requirements.txt         # Python environment requirements
├── run.sh                   # Reproducible one-command pipeline
└── README.md
```

---

## Quick Start

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Add datasets
Place all raw CSV and ARFF dataset files into:

```
project_data/
```

A small testing file (`sample_data.csv`) is included to allow the pipeline to run without large datasets.

---

### 3. Build the unified dataset
This step performs URL ingestion, lexical feature extraction, structured dataset loading, schema normalization,
deduplication, imputation, and optional scaling/SMOTE.

```bash
python src/build_phishing_dataset.py     --raw project_data/Phishing_URLs.csv project_data/top_1m.csv     --structured project_data/Phishing_Legitimate_full.csv     --out_dir out --do_scale --do_smote
```

Outputs (saved to `out/`):
- unified_clean.csv  
- X.npy / y.npy  
- prep_meta.json  

---

### 4. Train and evaluate models

```bash
python src/evaluate_models.py
```

This produces:
- Confusion matrices  
- Performance charts  
- Efficiency comparisons  
- Model metrics (AUC, F1, precision, recall, time, memory)

Copy the final outputs to:

```
reports/figures/
reports/results/
```

---

## Features & Models

### Extracted Features (25 total)
- URL length, entropy  
- Digit/symbol counts  
- Slash/dot/hyphen counts  
- TLD length  
- Subdomain count  
- Sensitive keyword flags  
- HTTPS flag  
- IP address detection  
- And more…

### Models Evaluated
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- LightGBM  

---

## Reproducibility

This repository includes:
- Complete preprocessing + training scripts  
- Full dataset provenance  
- Ethical statement  
- Environment specification  
- One-command `run.sh` execution  
- All final results and figures  

Clone and reproduce:

```bash
git clone https://github.com/<your-username>/phishing-url-detection.git
cd phishing-url-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

---

## Ethics & Data Usage

See `metadata/ETHICS.md` and `metadata/DATA_README.md` for:
- Data licensing  
- No-PII guarantees  
- Harm minimization  
- Misuse prevention  
- False-positive secret detection notice  

---

## Citation

```
Nirvik KC, "Hybrid Machine Learning-based Phishing URL Detection,"
CSC786: Special Topics in AI, Dakota State University, 2025.
```
