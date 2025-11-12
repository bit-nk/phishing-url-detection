# Workflow Summary (from CSC786_Ethics_Demo_ST.ipynb)

1. Inspect datasets and licenses (metadata/DATA_README.md)
2. Run feature extraction on raw URLs (src/build_phishing_dataset.py)
3. Merge structured datasets and clean/scale (script handles ARFF/CSV)
4. Save X.npy and y.npy and prep_meta.json for reproducibility
5. Evaluate models with evaluate_models.py and export figures and CSVs
