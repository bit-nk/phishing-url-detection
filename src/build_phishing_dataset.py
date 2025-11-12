import argparse
import os
import re
import math
import json
from typing import List, Dict, Tuple
from urllib.parse import urlparse
from collections import Counter

import numpy as np
import pandas as pd

#optional imports (only used if flags are enabled)
try:
    from scipy.io import arff as scipy_arff
except Exception:
    scipy_arff = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
except Exception:
    pass

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple columns share the same name (after renaming), merge them into one:
    - For numeric columns: take the first non-null (falling back to later columns)
    - For object/bool: same coalesce behavior
    Keeps column order stable using first occurrence.
    """
    cols = df.columns.tolist()
    seen = {}
    keep_cols = []
    for i, c in enumerate(cols):
        if c not in seen:
            seen[c] = [i]
            keep_cols.append(c)
        else:
            seen[c].append(i)

    out = pd.DataFrame(index=df.index)
    for c in keep_cols:
        idxs = seen[c]
        if len(idxs) == 1:
            out[c] = df.iloc[:, idxs[0]]
        else:
            #coalesce across duplicates (left-to-right first non-null)
            base = df.iloc[:, idxs[0]].copy()
            for j in idxs[1:]:
                mask = base.isna()
                if mask.any():
                    base[mask] = df.iloc[:, j][mask]
            out[c] = base
    return out



#0)Master schema & mappings

MASTER_FEATURES = [
    #lexical / counts
    "url_length","num_dots","num_dashes","num_underscores","num_slashes",
    "num_question_marks","num_equal_signs","num_ampersands","num_hashes",
    "num_percent","num_digits","special_char_count",
    #binary flags & derived
    "has_ip_address","has_at_symbol","has_prefix_suffix","num_subdomains","tld_length",
    "url_entropy","domain_length","path_length","query_length","tld_in_param",
    "https_flag","contains_sensitive_word","percentage_numeric_chars",
    #label last
    "label"
]

SUSPICIOUS_KEYWORDS = [
    "login","signin","verify","update","secure","bank","account","webscr",
    "confirm","pay","paypal","billing","password","credential","unlock",
    "support","alert","limited","suspend","invoice","wallet"
]


STRUCTURED_MAPPING = {
    #URL length
    "UrlLength":"url_length","URL_Length":"url_length","length_url":"url_length","url_length":"url_length",
    #counts / symbols
    "NumDots":"num_dots","dot_count":"num_dots","num_dots_url":"num_dots","num_dots_dom":"num_dots",
    "NumDash":"num_dashes","num_hyph_url":"num_dashes","num_hyph_dom":"num_dashes",
    "NumUnderscore":"num_underscores","num_underline_url":"num_underscores",
    "PathLevel":"num_slashes","num_slash_url":"num_slashes",
    "num_questionmark_url":"num_question_marks","num_questionmark_param":"num_question_marks",
    "num_equal_url":"num_equal_signs","num_equal_param":"num_equal_signs",
    "NumAmpersand":"num_ampersands","num_and_url":"num_ampersands","num_and_param":"num_ampersands",
    "NumHash":"num_hashes","hashtag_url":"num_hashes",
    "NumPercent":"num_percent","num_percent_url":"num_percent","num_percent_path":"num_percent","num_percent_param":"num_percent",
    "NumNumericChars":"num_digits","number_of_digits":"num_digits",
    "special_char_count":"special_char_count",
    #flags/derived
    "IpAddress":"has_ip_address","having_IP_Address":"has_ip_address","has_ip_address":"has_ip_address","dom_in_ip":"has_ip_address",
    "AtSymbol":"has_at_symbol","having_At_Symbol":"has_at_symbol","at_sign_url":"has_at_symbol",
    "Prefix_Suffix":"has_prefix_suffix","prefix_suffix_flag":"has_prefix_suffix",
    "SubdomainLevel":"num_subdomains","having_Sub_Domain":"num_subdomains","subdomain_count":"num_subdomains",
    "tld_length":"tld_length","num_tld_url":"tld_length",
    "url_entropy":"url_entropy",
    "HostnameLength":"domain_length","length_dom":"domain_length",
    "PathLength":"path_length","length_path":"path_length",
    "QueryLength":"query_length","length_param":"query_length",
    "tld_in_param":"tld_in_param",
    "https_flag":"https_flag","HTTPS_token":"https_flag","NoHttps":"https_flag","SSLfinal_State":"https_flag",
    "RandomString":"contains_sensitive_word",
    "NumSensitiveWords":"contains_sensitive_word",
    "percentage_numeric_chars":"percentage_numeric_chars",
    #labels
    "CLASS_LABEL":"label","Result":"label","ClassLabel":"label","phishing":"label","label":"label","Type":"label","type":"label"
}

LABEL_POS_VALUES = {"phishing","malicious","bad","1","true","yes"}
LABEL_NEG_VALUES = {"benign","legitimate","good","0","false","no"}



#1) Utils

def read_csv_safe(path: str) -> pd.DataFrame:
    for enc in (None, "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=None if enc is None else enc)
        except Exception:
            continue
    raise RuntimeError(f"Could not read CSV: {path}")

def read_arff_safe(path: str) -> pd.DataFrame:
    if scipy_arff is None:
        raise RuntimeError("scipy not available to read ARFF")
    data = scipy_arff.loadarff(path)
    df = pd.DataFrame(data[0])
    #decode bytes columns
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            try:
                df[c] = df[c].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
            except Exception:
                pass
    return df

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def is_ip_address(host: str) -> bool:
    if not host:
        return False
    #IPv4
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host):
        parts = host.split(".")
        return all(0 <= int(p) <= 255 for p in parts)
    #simple IPv6 check
    if ":" in host and re.fullmatch(r"[0-9a-fA-F:]+", host):
        return True
    return False

def count_subdomains(host: str) -> int:
    if not host:
        return 0
    parts = host.split(".")
    return max(len(parts) - 2, 0)

def tld_length_from_host(host: str) -> int:
    if not host:
        return 0
    parts = host.split(".")
    if len(parts) < 2:
        return 0
    return len(parts[-1])

def contains_any_keyword(s: str, words) -> int:
    if not s:
        return 0
    low = s.lower()
    return int(any(w in low for w in words))

def pct_numeric_chars(s: str) -> float:
    if not s:
        return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / len(s)

def extract_features_from_url(u: str) -> dict:
    try:
        u = str(u).strip()
        parsed = urlparse(u if re.match(r"^\w+://", u) else "http://" + u)
        host = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        full = parsed.geturl()

        feats = {
            "url": u,
            "url_length": len(full),
            "num_dots": full.count("."),
            "num_dashes": full.count("-"),
            "num_underscores": full.count("_"),
            "num_slashes": full.count("/"),
            "num_question_marks": full.count("?"),
            "num_equal_signs": full.count("="),
            "num_ampersands": full.count("&"),
            "num_hashes": full.count("#"),
            "num_percent": full.count("%"),
            "num_digits": sum(ch.isdigit() for ch in full),
            "special_char_count": sum(ch in r'~!@#$%^&*()_+-={}[]|:;"\'<>,.?/' for ch in full),
            "has_ip_address": int(is_ip_address(host)),
            "has_at_symbol": int("@" in full),
            "has_prefix_suffix": int("-" in host),
            "num_subdomains": count_subdomains(host),
            "tld_length": tld_length_from_host(host),
            "url_entropy": shannon_entropy(full),
            "domain_length": len(host),
            "path_length": len(path),
            "query_length": len(query),
            "tld_in_param": int(any(('.' in kv and kv.split('.')[-1].isalpha() and len(kv.split('.')[-1]) <= 6)
                                   for kv in query.split("&"))) if query else 0,
            "https_flag": int(parsed.scheme.lower() == "https"),
            "contains_sensitive_word": contains_any_keyword(full, SUSPICIOUS_KEYWORDS),
            "percentage_numeric_chars": pct_numeric_chars(full),
        }
        return feats
    except Exception:
        #return NaNs if parsing fails
        out = {k: np.nan for k in MASTER_FEATURES if k != "label"}
        out["url"] = u
        return out

def normalize_label_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    return np.where(s.isin(LABEL_POS_VALUES), 1,
           np.where(s.isin(LABEL_NEG_VALUES), 0,
           np.where(s.str.contains("phish"), 1,
           np.where(s.str.contains("legit|benign|good"), 0, np.nan)))).astype(float)



#2) Raw URL ingestion + features

def load_raw_urls(raw_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in raw_paths:
        if not os.path.isfile(p):
            print(f"[warn] Raw file not found: {p}")
            continue
        df = read_csv_safe(p)
                #heuristic: find url/domain column
        url_col = None
        for c in df.columns:
            if c.lower() in ("url","domain","site","link"):
                url_col = c
                break
        if url_col is None:
            #common top-1m.csv: two columns, second is domain
            if df.shape[1] >= 2:
                url_col = df.columns[1]
            else:
                print(f"[warn] No URL-like column in {p}; skipping")
                continue

        df["url"] = df[url_col].astype(str)

        #label column?
        lab_col = None
        for c in df.columns:
            if c.lower() in ("type","label","class","target"):
                lab_col = c
                break
        if lab_col is not None:
            df["label"] = normalize_label_series(df[lab_col])
        else:
            #special-case top 1m: assume legit
            if "top" in os.path.basename(p).lower():
                df["label"] = 0
            else:
                df["label"] = np.nan
        frames.append(df[["url","label"]])

    if not frames:
        return pd.DataFrame(columns=["url","label"])
    raw = pd.concat(frames, ignore_index=True).dropna(subset=["url"])
    raw["url"] = raw["url"].str.strip()
    raw = raw[raw["url"]!=""].drop_duplicates("url").reset_index(drop=True)
    return raw


def extract_features_for_dataframe(raw_df: pd.DataFrame, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Extract features in chunks to handle very large files without blowing memory/time.
    """
    if raw_df.empty:
        return raw_df
    chunks = []
    for start in range(0, len(raw_df), chunk_size):
        end = min(start+chunk_size, len(raw_df))
        sub = raw_df.iloc[start:end].copy()
        feats = [extract_features_from_url(u) for u in sub["url"].tolist()]
        feat_df = pd.DataFrame(feats)
        merged = sub.merge(feat_df, on="url", how="left")
        chunks.append(merged)
        print(f"[info] Extracted features for rows {start}..{end-1}")
    out = pd.concat(chunks, ignore_index=True)
    return out



#3) Structured dataset ingestion

def load_structured(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[warn] Structured file not found: {p}")
            continue
        try:
            if p.lower().endswith(".csv"):
                df = read_csv_safe(p)
            elif p.lower().endswith(".arff"):
                df = read_arff_safe(p)
            else:
                print(f"[warn] Unsupported structured format: {p}")
                continue
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}")
            continue

        #1) Rename to unified names
        rename_map = {c: STRUCTURED_MAPPING.get(c, c) for c in df.columns}
        df = df.rename(columns=rename_map)

        #2) Coalesce any duplicate columns created by renaming
        df = coalesce_duplicate_columns(df)

        #3) Normalize label if present
        if "label" in df.columns:
            if df["label"].dtype == object:
                try:
                    df["label"] = df["label"].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
                except Exception:
                    pass
            df["label"] = normalize_label_series(df["label"])
        else:
            df["label"] = np.nan

        #4) Keep only unified columns + url
        keep = set(MASTER_FEATURES) | {"url"}
        inter_cols = [c for c in df.columns if c in keep]
        if not inter_cols:
            print(f"[warn] No intersecting columns found in {p} after renaming; skipping")
            continue
        df = df[inter_cols]

        frames.append(df)
        print(f"[info] Loaded structured file: {p} shape={df.shape}")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = coalesce_duplicate_columns(merged)

    #Drop exact duplicate rows
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged



#4) Merge, clean, scale, SMOTE

def merge_and_clean(raw_feats: pd.DataFrame, structured_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not raw_feats.empty:
        frames.append(raw_feats)
    if not structured_df.empty:
        frames.append(structured_df)
    if not frames:
        return pd.DataFrame(columns=["url"] + MASTER_FEATURES)

    all_df = pd.concat(frames, ignore_index=True)

    #Ensure all needed columns exist
    for col in MASTER_FEATURES:
        if col not in all_df.columns:
            all_df[col] = np.nan
    if "url" not in all_df.columns:
        all_df["url"] = ""

    #Dtypes + imputation
    num_cols = [c for c in MASTER_FEATURES if c != "label"]
    for c in num_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    #label as float -> int later
    all_df["label"] = pd.to_numeric(all_df["label"], errors="coerce")

    #Drop rows without label
    before = len(all_df)
    all_df = all_df.dropna(subset=["label"])
    print(f"[info] Dropped {before - len(all_df)} rows without label")

    #Impute numeric with median
    med = all_df[num_cols].median(numeric_only=True)
    all_df[num_cols] = all_df[num_cols].fillna(med)

    #Clip extreme values (optional small safety)
    all_df[num_cols] = all_df[num_cols].clip(lower=all_df[num_cols].quantile(0.001),
                                             upper=all_df[num_cols].quantile(0.999),
                                             axis=1)

    all_df["label"] = all_df["label"].astype(int)
    all_df = all_df.drop_duplicates(subset=["url","label"] + num_cols).reset_index(drop=True)
    return all_df


def scale_and_balance(df, do_scale, do_smote, random_state=42):
    X = df[[c for c in MASTER_FEATURES if c != "label"]].copy()
    y = df["label"].copy()
    meta = {}
    if do_scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[:] = scaler.fit_transform(X)
        meta["scaler"] = scaler
    #return raw X,y here; SMOTE will be done after train/test split
    return X, y, meta



#5) Optional quick baselines

def train_quick_baselines(X, y, out_dir, random_state=42, use_smote=False):

    #Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    #Apply SMOTE only on training set
    if use_smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)  # no n_jobs argument
        X_train, y_train = sm.fit_resample(X_train, y_train)

    models = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state)
    }

    reports = {}
    os.makedirs(out_dir, exist_ok=True)

    for name, model in models.items():
        print(f"[model:{name}] training on {len(y_train)} samples...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None

        rep = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        reports[name] = {
            "auc": auc,
            "accuracy": rep["accuracy"],
            "f1_macro": rep["macro avg"]["f1-score"],
            "f1_weighted": rep["weighted avg"]["f1-score"],
            "confusion_matrix": cm.tolist()
        }
        print(f"[model:{name}] AUC={auc}  ACC={rep['accuracy']:.4f}")

    with open(os.path.join(out_dir, "model_reports.json"), "w") as f:
        json.dump(reports, f, indent=2)
    print(f"[info] Saved model reports to {os.path.join(out_dir, 'model_reports.json')}")




#6) Main

def main():
    ap = argparse.ArgumentParser(description="Build unified phishing URL dataset with feature extraction + preprocessing.")
    ap.add_argument("--raw", nargs="*", default=[], help="Raw URL CSVs (e.g., Phishing_URLs.csv, URL_dataset.csv, top_1m.csv)")
    ap.add_argument("--structured", nargs="*", default=[], help="Structured CSV/ARFF files with features")
    ap.add_argument("--out_dir", default="./out", help="Directory to write outputs")
    ap.add_argument("--chunk", type=int, default=100000, help="Feature extraction chunk size for raw URLs")
    ap.add_argument("--do_scale", action="store_true", help="Apply StandardScaler to features")
    ap.add_argument("--do_smote", action="store_true", help="Apply SMOTE to balance classes (requires imblearn)")
    ap.add_argument("--train_models", action="store_true", help="Train quick baseline models and save metrics")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    #A) RAW URLs -> features
    if args.raw:
        print("[step] Loading raw URL files...")
        raw_df = load_raw_urls(args.raw)
        print(f"[info] Raw URLs loaded: {len(raw_df)}")
        if not raw_df.empty:
            print("[step] Extracting lexical features from raw URLs...")
            raw_feats = extract_features_for_dataframe(raw_df, chunk_size=args.chunk)
            #Save intermediate
            raw_feats.to_csv(os.path.join(args.out_dir, "raw_url_features.csv"), index=False)
            print(f"[info] Wrote {os.path.join(args.out_dir,'raw_url_features.csv')}  rows={len(raw_feats)}")
        else:
            raw_feats = pd.DataFrame()
            print("[info] No raw URLs found/parsed.")
    else:
        raw_feats = pd.DataFrame()

    #B) Structured datasets
    if args.structured:
        print("[step] Loading structured feature datasets...")
        structured_df = load_structured(args.structured)
        print(f"[info] Structured rows: {len(structured_df)}")
        structured_df.to_csv(os.path.join(args.out_dir, "structured_renamed.csv"), index=False)
    else:
        structured_df = pd.DataFrame()

    #C) Merge + clean
    print("[step] Merging and cleaning...")
    merged = merge_and_clean(raw_feats, structured_df)
    merged.to_csv(os.path.join(args.out_dir, "unified_clean.csv"), index=False)
    print(f"[info] Wrote unified dataset: {os.path.join(args.out_dir,'unified_clean.csv')}  rows={len(merged)}")

    #D) Scale / SMOTE
    print("[step] Preparing X,y with optional scaling/SMOTE...")
    X, y, meta = scale_and_balance(merged, do_scale=args.do_scale, do_smote=args.do_smote, random_state=args.random_state)
    np.save(os.path.join(args.out_dir, "X.npy"), X.values)
    np.save(os.path.join(args.out_dir, "y.npy"), y.values)
    with open(os.path.join(args.out_dir, "prep_meta.json"), "w") as f:
        json.dump({"do_scale": args.do_scale, "do_smote": args.do_smote, "random_state": args.random_state}, f, indent=2)
    print(f"[info] Saved X.npy, y.npy, prep_meta.json in {args.out_dir}")

    #E) Optional quick baselines
    if args.train_models:
        print("[step] Training quick baselines...")
        train_quick_baselines(X, y, args.out_dir, random_state=args.random_state, use_smote=args.do_smote)


    print("[done] Pipeline complete.")


if __name__ == "__main__":
    main()
