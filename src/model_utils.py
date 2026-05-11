# =============================================================
# src/model_utils.py
# Model Loading & Prediction Utilities
# AI Personal Finance Analyzer
# =============================================================
# WHAT THIS FILE DOES:
# - Loads trained pipelines cleanly
# - Provides prediction interface for Streamlit app
# - Handles feature engineering for new transactions
# - Returns fraud probability and risk level
# =============================================================

import pandas as pd
import numpy as np
import joblib
import json
import os
import logging

# -------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# STEP 1 — Load model config
# -------------------------------------------------------------
def load_config(config_path: str = "models/model_config.json") -> dict:
    """
    Loads model configuration from JSON file.

    Args:
        config_path: path to config JSON file

    Returns:
        config dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"loaded config: model v{config['model_version']}")
    return config


# -------------------------------------------------------------
# STEP 2 — Load a specific pipeline
# -------------------------------------------------------------
def load_pipeline(model_name: str = "best",
                  models_dir: str = "models/") -> object:
    """
    Loads a trained sklearn Pipeline from disk.

    Args:
        model_name: 'best', 'xgboost', 'random_forest',
                    or 'logistic_regression'
        models_dir: folder containing saved pipelines

    Returns:
        loaded sklearn Pipeline object
    """
    if model_name == "best":
        filepath = os.path.join(models_dir,
                                "best_model_pipeline.pkl")
    else:
        filepath = os.path.join(
            models_dir, f"{model_name}_pipeline.pkl"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"pipeline not found at {filepath}"
        )

    pipeline = joblib.load(filepath)
    logger.info(f"loaded pipeline: {filepath}")

    return pipeline


# -------------------------------------------------------------
# STEP 3 — Engineer features for a single transaction
# this mirrors what we did in features.py
# but for a single new transaction from the UI
# -------------------------------------------------------------
def engineer_features(transaction: dict) -> pd.DataFrame:
    """
    Engineers features for a single new transaction.

    This mirrors the feature engineering pipeline
    but for real-time prediction from the Streamlit UI.

    Args:
        transaction: dict with raw transaction fields

    Returns:
        DataFrame with all engineered features ready
        for model prediction
    """
    # convert to dataframe
    df = pd.DataFrame([transaction])

    # ── basic balance features ───────────────────────────────
    df['balance_diff_orig'] = (
        df['oldbalanceOrg'] - df['newbalanceOrig']
    )
    df['balance_diff_dest'] = (
        df['newbalanceDest'] - df['oldbalanceDest']
    )
    df['orig_balance_zero'] = (
        df['newbalanceOrig'] == 0
    ).astype(int)
    df['dest_balance_zero'] = (
        df['oldbalanceDest'] == 0
    ).astype(int)
    df['amount_to_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )

    # ── time features ────────────────────────────────────────
    df['hour_of_day']        = df['step'] % 24
    df['day_of_simulation']  = df['step'] // 24
    df['is_late_night']      = (
        (df['hour_of_day'] >= 23) |
        (df['hour_of_day'] <= 5)
    ).astype(int)

    # ── transaction risk features ────────────────────────────
    df['is_high_risk_type']  = (
        df['type'].isin([1, 4])
    ).astype(int)
    df['is_transfer']        = (df['type'] == 4).astype(int)
    df['is_cash_out']        = (df['type'] == 1).astype(int)

    # ── balance mismatch features ────────────────────────────
    df['expected_new_balance_orig'] = (
        df['oldbalanceOrg'] - df['amount']
    )
    df['balance_mismatch_orig'] = abs(
        df['newbalanceOrig'] - df['expected_new_balance_orig']
    )
    df['expected_new_balance_dest'] = (
        df['oldbalanceDest'] + df['amount']
    )
    df['balance_mismatch_dest'] = abs(
        df['newbalanceDest'] - df['expected_new_balance_dest']
    )
    df['has_balance_mismatch'] = (
        (df['balance_mismatch_orig'] > 0) |
        (df['balance_mismatch_dest'] > 0)
    ).astype(int)

    # ── amount features ──────────────────────────────────────
    df['amount_zscore']        = 0.0  # neutral for single tx
    df['is_large_transaction'] = (
        df['amount'] > 200000
    ).astype(int)
    df['is_round_amount']      = (
        df['amount'] % 1000 == 0
    ).astype(int)
    df['log_amount']           = np.log1p(df['amount'])

    # ── receiver risk features ───────────────────────────────
    df['receiver_was_empty']    = (
        df['oldbalanceDest'] == 0
    ).astype(int)
    df['large_amount_to_empty'] = (
        (df['receiver_was_empty'] == 1) &
        (df['is_large_transaction'] == 1)
    ).astype(int)

    # load config to get exact feature order
    config = load_config()
    feature_cols = config['features']

    return df[feature_cols]


# -------------------------------------------------------------
# STEP 4 — Make prediction
# -------------------------------------------------------------
def predict_fraud(transaction: dict,
                  model_name: str = "best",
                  models_dir: str = "models/") -> dict:
    """
    Makes a fraud prediction for a single transaction.

    Returns probability, binary prediction, and
    a human-readable risk level for the UI.

    Risk levels:
    - LOW:      probability < 0.3
    - MEDIUM:   probability 0.3 - 0.7
    - HIGH:     probability 0.7 - 0.9
    - CRITICAL: probability > 0.9

    Args:
        transaction: raw transaction dict from UI
        model_name: which model to use
        models_dir: folder with saved pipelines

    Returns:
        dict with prediction results
    """
    # load pipeline
    pipeline = load_pipeline(model_name, models_dir)

    # engineer features
    X = engineer_features(transaction)

    # get prediction
    fraud_prob    = pipeline.predict_proba(X)[0][1]
    is_fraud      = pipeline.predict(X)[0]

    # determine risk level
    if fraud_prob < 0.3:
        risk_level = "LOW"
        risk_color = "green"
    elif fraud_prob < 0.7:
        risk_level = "MEDIUM"
        risk_color = "orange"
    elif fraud_prob < 0.9:
        risk_level = "HIGH"
        risk_color = "red"
    else:
        risk_level = "CRITICAL"
        risk_color = "darkred"

    return {
        'fraud_probability': float(fraud_prob),
        'is_fraud':          int(is_fraud),
        'risk_level':        risk_level,
        'risk_color':        risk_color,
        'fraud_pct':         f"{fraud_prob * 100:.2f}%"
    }


# -------------------------------------------------------------
# STEP 5 — Batch prediction for uploaded CSV
# -------------------------------------------------------------
def predict_batch(df: pd.DataFrame,
                  model_name: str = "best",
                  models_dir: str = "models/") -> pd.DataFrame:
    """
    Makes fraud predictions for an entire uploaded CSV.

    This is what the Streamlit UI calls when a user
    uploads their bank statement CSV.

    Args:
        df: raw transaction dataframe from user upload
        model_name: which model to use
        models_dir: folder with saved pipelines

    Returns:
        dataframe with fraud_probability and risk_level
        columns added
    """
    logger.info(f"running batch prediction on {len(df):,} rows...")

    pipeline = load_pipeline(model_name, models_dir)
    config   = load_config()

    # encode transaction type if it's text
    type_mapping = config['transaction_type_mapping']
    if df['type'].dtype == object:
        df['type'] = df['type'].map(type_mapping).fillna(0)

    # engineer all features
    results = []
    for _, row in df.iterrows():
        try:
            transaction = row.to_dict()
            result = predict_fraud(
                transaction, model_name, models_dir
            )
            results.append(result)
        except Exception as e:
            results.append({
                'fraud_probability': 0.0,
                'is_fraud':          0,
                'risk_level':        'UNKNOWN',
                'risk_color':        'gray',
                'fraud_pct':         '0.00%'
            })

    # add results to dataframe
    results_df = pd.DataFrame(results)
    df = df.reset_index(drop=True)
    df['fraud_probability'] = results_df['fraud_probability']
    df['risk_level']        = results_df['risk_level']
    df['is_fraud_predicted'] = results_df['is_fraud']

    logger.info(f"batch prediction complete")
    logger.info(
        f"flagged as fraud: {df['is_fraud_predicted'].sum():,}"
    )

    return df