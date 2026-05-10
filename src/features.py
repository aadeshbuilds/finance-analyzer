# =============================================================
# src/features.py
# Advanced Feature Engineering Pipeline
# AI Personal Finance Analyzer
# =============================================================
# WHAT THIS FILE DOES:
# - Takes cleaned data from preprocessing pipeline
# - Engineers advanced features for ML model
# - Selects the most important features
# - Saves feature-engineered data ready for training
# =============================================================

import pandas as pd
import numpy as np
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
# FEATURE 1 — Time-based features
# 'step' represents hours since simulation started
# -------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the step column.

    In real bank data this would come from timestamps.
    Here step = hour number, so we extract hour of day
    and day of week as proxy features.

    Args:
        df: cleaned dataframe

    Returns:
        dataframe with time features added
    """
    logger.info("adding time-based features...")

    # hour of day (0-23) — fraud often happens late night
    df['hour_of_day'] = df['step'] % 24

    # day number in simulation
    df['day_of_simulation'] = df['step'] // 24

    # is it a late night transaction? (11pm - 5am)
    df['is_late_night'] = (
        (df['hour_of_day'] >= 23) | 
        (df['hour_of_day'] <= 5)
    ).astype(int)

    logger.info("added: hour_of_day, day_of_simulation, is_late_night")

    return df


# -------------------------------------------------------------
# FEATURE 2 — Transaction risk features
# based on type of transaction
# -------------------------------------------------------------
def add_transaction_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates risk flags based on transaction type.

    From EDA we know ONLY CASH_OUT (1) and TRANSFER (4)
    have fraud. So we flag these as high risk.

    Args:
        df: dataframe with time features

    Returns:
        dataframe with risk features added
    """
    logger.info("adding transaction risk features...")

    # high risk transaction types
    # CASH_OUT = 1, TRANSFER = 4 (from our label encoding)
    df['is_high_risk_type'] = (
        df['type'].isin([1, 4])
    ).astype(int)

    # is it a transfer specifically?
    df['is_transfer'] = (df['type'] == 4).astype(int)

    # is it a cash out specifically?
    df['is_cash_out'] = (df['type'] == 1).astype(int)

    logger.info("added: is_high_risk_type, is_transfer, is_cash_out")

    return df


# -------------------------------------------------------------
# FEATURE 3 — Balance mismatch features
# this is the most powerful fraud signal
# -------------------------------------------------------------
def add_balance_mismatch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects suspicious balance manipulation.

    In real fraud: the reported balance change doesn't
    match what the transaction amount should produce.
    Fraudsters manipulate balance fields to hide activity.

    Args:
        df: dataframe with risk features

    Returns:
        dataframe with balance mismatch features added
    """
    logger.info("adding balance mismatch features...")

    # expected new balance = old balance - amount
    # if actual differs → something suspicious
    df['expected_new_balance_orig'] = (
        df['oldbalanceOrg'] - df['amount']
    )

    df['balance_mismatch_orig'] = abs(
        df['newbalanceOrig'] - df['expected_new_balance_orig']
    )

    # same check for destination account
    df['expected_new_balance_dest'] = (
        df['oldbalanceDest'] + df['amount']
    )

    df['balance_mismatch_dest'] = abs(
        df['newbalanceDest'] - df['expected_new_balance_dest']
    )

    # flag large mismatches — strong fraud signal
    df['has_balance_mismatch'] = (
        (df['balance_mismatch_orig'] > 0) | 
        (df['balance_mismatch_dest'] > 0)
    ).astype(int)

    logger.info("added: balance_mismatch_orig, balance_mismatch_dest, "
                "has_balance_mismatch")

    return df


# -------------------------------------------------------------
# FEATURE 4 — Amount-based features
# unusual amounts are suspicious
# -------------------------------------------------------------
def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from transaction amount patterns.

    Fraud often involves unusually large amounts or
    suspiciously round numbers.

    Args:
        df: dataframe with balance features

    Returns:
        dataframe with amount features added
    """
    logger.info("adding amount-based features...")

    # how far is this amount from the mean?
    # called z-score normalization
    mean_amount = df['amount'].mean()
    std_amount  = df['amount'].std()

    df['amount_zscore'] = (
        (df['amount'] - mean_amount) / std_amount
    )

    # is this an unusually large transaction?
    # more than 2 standard deviations above mean
    df['is_large_transaction'] = (
        df['amount_zscore'] > 2
    ).astype(int)

    # is the amount a round number?
    # fraudsters often use round amounts like 10000.00
    df['is_round_amount'] = (
        df['amount'] % 1000 == 0
    ).astype(int)

    # log transform of amount
    # reduces the effect of extreme outliers
    df['log_amount'] = np.log1p(df['amount'])

    logger.info("added: amount_zscore, is_large_transaction, "
                "is_round_amount, log_amount")

    return df


# -------------------------------------------------------------
# FEATURE 5 — Receiver risk features
# mule accounts show specific patterns
# -------------------------------------------------------------
def add_receiver_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies suspicious receiver account patterns.

    Money mule accounts (used to receive stolen funds)
    typically start with zero balance and receive
    large amounts suddenly.

    Args:
        df: dataframe with amount features

    Returns:
        dataframe with receiver risk features added
    """
    logger.info("adding receiver risk features...")

    # receiver had zero balance before receiving
    # classic mule account pattern
    df['receiver_was_empty'] = (
        df['oldbalanceDest'] == 0
    ).astype(int)

    # large amount into empty account — very suspicious
    df['large_amount_to_empty'] = (
        (df['receiver_was_empty'] == 1) & 
        (df['is_large_transaction'] == 1)
    ).astype(int)

    logger.info("added: receiver_was_empty, large_amount_to_empty")

    return df


# -------------------------------------------------------------
# FEATURE SELECTION — keep only the best features
# -------------------------------------------------------------
def select_features(df: pd.DataFrame) -> tuple:
    """
    Selects the final feature set for ML training.

    Removes raw columns that have been replaced by
    engineered versions. Returns X (features) and
    y (target) separately — ready for model training.

    Args:
        df: fully feature-engineered dataframe

    Returns:
        tuple of (X DataFrame, y Series)
    """
    logger.info("selecting final feature set...")

    # columns to drop — either leaky or replaced
    # by engineered versions
    cols_to_drop = [
        'isFraud',      # this is our target, not a feature
        'step',         # replaced by hour_of_day, day_of_simulation
        'type',         # replaced by is_high_risk_type etc
    ]

    # our final features
    feature_cols = [col for col in df.columns 
                    if col not in cols_to_drop]

    X = df[feature_cols]
    y = df['isFraud']

    logger.info(f"final feature count: {len(feature_cols)}")
    logger.info(f"features: {feature_cols}")
    logger.info(f"target distribution: {y.value_counts().to_dict()}")

    return X, y


# -------------------------------------------------------------
# MASTER FUNCTION
# -------------------------------------------------------------
def run_feature_pipeline(
    input_path: str,
    output_path: str
) -> tuple:
    """
    Runs the complete feature engineering pipeline.

    Args:
        input_path: path to cleaned CSV
        output_path: path to save feature-engineered CSV

    Returns:
        tuple of (X, y) ready for model training
    """
    logger.info("=" * 50)
    logger.info("starting feature engineering pipeline")
    logger.info("=" * 50)

    # load cleaned data
    df = pd.read_csv(input_path)
    logger.info(f"loaded cleaned data: {df.shape}")

    # run all feature engineering steps
    df = add_time_features(df)
    df = add_transaction_risk_features(df)
    df = add_balance_mismatch_features(df)
    df = add_amount_features(df)
    df = add_receiver_risk_features(df)

    # save feature engineered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"saved feature-engineered data: {df.shape}")

    # select features for training
    X, y = select_features(df)

    logger.info("=" * 50)
    logger.info("feature engineering pipeline complete!")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info("=" * 50)

    return X, y


# -------------------------------------------------------------
# Run directly from terminal
# python src/features.py
# -------------------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH  = "data/processed/transactions_clean.csv"
    OUTPUT_PATH = "data/processed/transactions_features.csv"

    X, y = run_feature_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH
    )

    print("\nfeature engineering complete!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nfeatures created:")
    for col in X.columns:
        print(f"  → {col}")