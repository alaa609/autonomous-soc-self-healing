from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONTEXT_COLS = [
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
]


COLUMN_ALIASES = {
    "src ip": "Source IP",
    "source ip": "Source IP",
    "src_ip": "Source IP",
    "source_ip": "Source IP",
    "dst ip": "Destination IP",
    "dest ip": "Destination IP",
    "destination ip": "Destination IP",
    "dst_ip": "Destination IP",
    "destination_ip": "Destination IP",
    "src port": "Source Port",
    "source port": "Source Port",
    "src_port": "Source Port",
    "sport": "Source Port",
    "dst port": "Destination Port",
    "dest port": "Destination Port",
    "destination port": "Destination Port",
    "dst_port": "Destination Port",
    "dport": "Destination Port",
    "protocol": "Protocol",
    "proto": "Protocol",
    "timestamp": "Timestamp",
    "time": "Timestamp",
    "total fwd packets": "Total Packets Forward",
    "total packets forward": "Total Packets Forward",
    "fwd packets": "Total Packets Forward",
    "total backward packets": "Total Packets Backward",
    "total bwd packets": "Total Packets Backward",
    "total packets backward": "Total Packets Backward",
    "bwd packets": "Total Packets Backward",
    "total length of fwd packets": "Total Bytes Forward",
    "fwd bytes": "Total Bytes Forward",
    "total bytes forward": "Total Bytes Forward",
    "total length of bwd packets": "Total Bytes Backward",
    "bwd bytes": "Total Bytes Backward",
    "total bytes backward": "Total Bytes Backward",
    "flow duration": "Flow Duration",
    "fwd iat mean": "Flow IAT Mean",
    "flow iat mean": "Flow IAT Mean",
    "flow iat std": "Flow IAT Std",
    "flow iat max": "Flow IAT Max",
    "flow iat min": "Flow IAT Min",
    "label": "Label",
    "attack": "Label",
    "class": "Label",
}


def normalize_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    renamed = {}
    existing_cols = set(df.columns)
    for col in df.columns:
        key = col.strip().lower()
        target_name = COLUMN_ALIASES.get(key)
        if target_name and target_name != col and target_name not in existing_cols and target_name not in renamed.values():
            renamed[col] = target_name
        else:
            renamed[col] = col.strip()
    return df.rename(columns=renamed)


def load_telemetry_data(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    
    if isinstance(source, pd.DataFrame):
        return normalize_team_columns(source.copy())

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use DataFrame, .csv, or .parquet.")

    return normalize_team_columns(df)


def prepare_soc_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    time_col: str = "Timestamp",
    window: int = 5,
):
    
    data = normalize_team_columns(df.copy())
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    resolved_target = None
    if target_col:
        if target_col in data.columns:
            resolved_target = target_col
        elif "Label" in data.columns:
            resolved_target = "Label"

    y = None
    if resolved_target is not None:
        y = data[resolved_target]
        work_df = data.drop(columns=[resolved_target])
    else:
        work_df = data

    resolved_time_col = time_col if time_col in work_df.columns else ("Timestamp" if "Timestamp" in work_df.columns else None)

   
    present_context = [c for c in CONTEXT_COLS if c in work_df.columns]
    context_df = work_df[present_context].copy() if present_context else pd.DataFrame(index=work_df.index)

    
    if resolved_time_col:
        ts_parsed = pd.to_datetime(work_df[resolved_time_col], errors="coerce")
        if ts_parsed.notna().any():
            work_df["_ts_sort"] = ts_parsed
            work_df = work_df.sort_values("_ts_sort", na_position="last").reset_index(drop=True)
            context_df = context_df.loc[work_df.index].reset_index(drop=True)
            if y is not None:
                y = y.loc[work_df.index].reset_index(drop=True)
            work_df.drop(columns=["_ts_sort"], inplace=True)

    
    drop_set = {"Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol"}
    if resolved_time_col:
        drop_set.add(resolved_time_col)
    cols_to_drop_from_X = [c for c in drop_set if c in work_df.columns]
    X = work_df.drop(columns=cols_to_drop_from_X).select_dtypes(include=[np.number]).copy()

    # Add rolling window features on key flow/packet counters
    rolling_candidates = [
        c for c in X.columns
        if any(k in c.lower() for k in ["bytes", "packets", "duration", "iat"])
    ]
    for col in rolling_candidates[:4]:
        X[f"{col}_roll_mean_{window}"] = X[col].rolling(window=window, min_periods=1).mean()
        X[f"{col}_roll_std_{window}"] = X[col].rolling(window=window, min_periods=1).std().fillna(0.0)

    # Fill missing values with median (fallback to 0.0)
    for col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(0.0 if pd.isna(med) else med)

    return X, y, context_df, X.columns.tolist()


def run_sequential_pipeline(
    source: str | Path | pd.DataFrame,
    target_col: str | None = "Label",
    time_col: str = "Timestamp",
    window: int = 5,
    contamination: float = 0.05,
    random_state: int = 42,
):
    
    df = load_telemetry_data(source)
    X, y, context_df, feature_cols = prepare_soc_features(
        df=df,
        target_col=target_col,
        time_col=time_col,
        window=window,
    )

    # Step 1: Isolation Forest anomaly detection
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso_model.fit(X)

    anomaly_score = -iso_model.decision_function(X)
    iso_pred = iso_model.predict(X)  # 1: normal, -1: anomaly
    is_anomaly = iso_pred == -1

    normal_mask = ~is_anomaly
    X_normal = X.loc[normal_mask]
    X_anomaly = X.loc[is_anomaly]

    results = {
        "X_features": X,
        "feature_cols": feature_cols,
        "context": context_df,
        "iso_model": iso_model,
        "anomaly_score": anomaly_score,
        "iso_pred": iso_pred,
        "normal_count": int(normal_mask.sum()),
        "anomaly_count": int(is_anomaly.sum()),
        "normal_context": context_df.loc[normal_mask],
        "rf_model": None,
        "scaler": None,
        "anomaly_results": None,
        "top_features": None,
    }

    
    if y is not None and y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y if y.value_counts().min() >= 2 else None,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train_scaled, y_train)

        y_test_pred = rf_model.predict(X_test_scaled)
        results["rf_model"] = rf_model
        results["scaler"] = scaler
        results["macro_f1"] = f1_score(y_test, y_test_pred, average="macro")
        results["classification_report"] = classification_report(y_test, y_test_pred)
        results["top_features"] = pd.Series(
            rf_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        if len(X_anomaly) > 0:
            X_anom_scaled = scaler.transform(X_anomaly)
            rf_pred = rf_model.predict(X_anom_scaled)
            rf_conf = rf_model.predict_proba(X_anom_scaled).max(axis=1)

            anom_table = context_df.loc[is_anomaly].copy()
            anom_table["anomaly_score"] = anomaly_score[is_anomaly]
            anom_table["attack_class"] = rf_pred
            anom_table["confidence"] = rf_conf
            results["anomaly_results"] = anom_table

    return results


if __name__ == "__main__":
   
    team_data_path = "team_telemetry.csv"

    out = run_sequential_pipeline(
        source=team_data_path,
        target_col="Label",  # Set to None if team data has no labels yet
        time_col="Timestamp",
        window=5,
        contamination=0.05,
        random_state=42,
    )

    print("Feature count:", len(out["feature_cols"]))
    print("Normal count:", out["normal_count"])
    print("Anomaly count (sent to RF):", out["anomaly_count"])
    if out["rf_model"] is not None:
        print("Macro F1:", round(out["macro_f1"], 4))
        print("\nTop 5 features:\n", out["top_features"].head(5))
        if out["anomaly_results"] is not None:
            print("\nAnomaly classification preview:\n", out["anomaly_results"].head())
