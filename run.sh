#!/bin/bash
set -e
echo "[1] Build dataset (feature extraction + merge)"
python src/build_phishing_dataset.py --raw project_data/sample_data.csv --out_dir out --do_scale --do_smote
echo "[2] Evaluate models"
python src/evaluate_models.py
echo "Done. Check ./out for outputs."
