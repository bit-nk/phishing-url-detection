# Phishing URL Detection — Reproducible Repo (Template)

This repository is a reproducible template for your Phishing-URL Detection project, following the Data Ethics & Reproducibility Workshop guidelines.
It is customized for the scripts you already have: `build_phishing_dataset.py` and `evaluate_models.py`.

## Structure (what to fill)
- `src/` : core scripts (data build + evaluation)
- `project_data/` : raw datasets (place your CSV/ARFF files here)
- `out/` : pipeline outputs (will be created by scripts; kept out of git)
- `reports/figures` and `reports/results` : final figures and result tables for the paper
- `metadata/` : DATA_README.md, ETHICS.md, LICENSE

## Quick start (example)
1. Create and activate a Python virtual environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Put your dataset files into `project_data/` (or keep the sample_data.csv for a small demo).

3. Run the data build (feature extraction + merge):
```bash
python src/build_phishing_dataset.py --raw project_data/sample_data.csv --out_dir out --do_scale --do_smote
```

4. Run model evaluation:
```bash
python src/evaluate_models.py
```

5. Results and figures will appear under `out/` and can be copied to `reports/` for publication.

## Notes
- This template follows the steps outlined in your `CSC786_Ethics_Demo_ST.ipynb` notebook: prepare data → extract features → merge & clean → scale & SMOTE → train baselines → evaluate efficiency & save figures.
- Replace `project_data/sample_data.csv` with your full datasets when ready.
