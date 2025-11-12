# Dataset Overview (`project_data/`)

This folder contains all **raw datasets** used to build and reproduce the phishing–legitimate URL detection pipeline.  
All datasets included are **public**, contain **no PII**, and are used strictly for academic research.

---

## **Included Datasets**

### 1. **Phishing_URLs.csv**
- Raw phishing URL list.
- Contains only URLs and labels (phishing vs legitimate).
- Used for lexical feature extraction.
- Public dataset commonly sourced from Kaggle.

---

### 2. **Phishing URLs.csv**
- Variant naming of a phishing URL dataset.
- Contains raw phishing URLs.
- Same usage as above.

---

### 3. **URL_dataset.csv**
- Mixed dataset containing phishing and legitimate URLs.
- Schema varies but includes raw URLs + labels.
- Used to diversify URL characteristics.

---

### 4. **URL dataset.csv**
- Another naming variant of a raw URL dataset.
- Contains mixed phishing and legitimate URLs.
- Used in lexical feature extraction and label normalization.

---

### 5. **Phishing_Legitimate_full.csv**
- Structured CSV containing extracted URL features.
- Feature categories include:
  - Symbol counts
  - Domain length
  - Path/query length
  - Shannon entropy
  - IP address flags
  - Sensitive keyword indicators
- Merged directly during structured preprocessing.

---

### 6. **Phishing_Legitimate_full.arff**
- ARFF version of the above dataset.
- Used for compatibility with ARFF-based ML datasets.
- Parsed automatically via SciPy.

---

### 7. **Training_Dataset.arff**
- Structured ARFF dataset with pre-engineered phishing features.
- Includes numeric and categorical predictors.
- Augments the structured feature inputs.

---

### 8. **Zieni_Dataset_for_phishing_detection.csv**
- Kaggle dataset published by Zieni.
- Contains balanced phishing and legitimate URLs.
- Includes raw URLs and some pre-derived attributes.
- Improves dataset variety and balance.

---

### 9. **url_features_extracted1.csv**
- Previously extracted lexical features from an earlier run.
- Useful for testing pipeline behavior without re-running extraction.
- Treated as structured input.

---

### 10. **top_1m.csv**
- Tranco Top 1 Million domains (legitimate-only baseline).
- Used to strengthen representation of legitimate class.
- All entries labeled `0`.
- Licensed under **CC BY 4.0**.

---

## **Reproducibility Notes**

The unified dataset is constructed by:

1. Loading raw URL lists  
2. Identifying URL columns and normalizing labels  
3. Extracting lexical features (length, entropy, symbols, subdomains, etc.)  
4. Loading structured CSV/ARFF datasets  
5. Renaming columns to a master schema  
6. Coalescing duplicate or overlapping fields  
7. Merging all datasets  
8. Dropping unlabeled rows  
9. Median-imputing missing numeric fields  
10. Optional: Standard scaling  
11. Optional: SMOTE balancing  
12. Saving:
    - `unified_clean.csv`
    - `X.npy` (model inputs)
    - `y.npy` (binary labels)
    - `prep_meta.json` (scaling + SMOTE + random state info)

This ensures fully reproducible preprocessing.

---

## Licensing & Ethics

- No dataset contains personal information or sensitive user data.  
- All datasets originate from public sources or academic repositories.
- Redistribution allowed under:
  - **Tranco Top 1M:** CC BY 4.0  
  - **Kaggle Datasets:** Public under Kaggle Terms of Service  
  - **Academic ARFF datasets:** Research/educational use  

Use of these datasets is strictly for **academic and non-commercial research**.

---

## Notes

- Do **not** modify original datasets—keep them exactly as downloaded.
- New datasets may be added if they contain URLs and labels and are publicly redistributable.

