# =============================================================
# src/preprocessing.py
# Data Cleaning & Preprocessing Pipeline
# AI Personal Finance Analyzer
# =============================================================
# WHAT THIS FILE DOES:
# - Loads raw transaction data
# - Samples it intelligently (keeps class balance)
# - Drops useless and leaky columns
# - Encodes categorical variables
# - Saves clean data to data/processed/
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import logging

# -------------------------------------------------------------
# Setup logging — production code always logs what it's doing
# instead of using print() everywhere
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# STEP 1 — Load raw data
# -------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV transaction data from the given filepath.

    Args:
        filepath: full path to the raw CSV file

    Returns:
        pandas DataFrame with raw data
    """
    logger.info(f"loading data from {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"data file not found at {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"loaded {len(df):,} rows and {df.shape[1]} columns")

    return df


# -------------------------------------------------------------
# STEP 2 — Smart sampling
# keeps fraud cases, samples normal cases
# this is called stratified sampling
# -------------------------------------------------------------
def sample_data(df: pd.DataFrame,
                normal_sample_size: int = 50000) -> pd.DataFrame:
    """
    Reduces dataset size while preserving ALL fraud cases.

    Why: 6.3M rows is too large for laptop training.
    Strategy: keep all fraud + sample normal transactions.
    This is called stratified sampling.

    Args:
        df: raw dataframe
        normal_sample_size: how many normal transactions to keep

    Returns:
        smaller but balanced-ish dataframe
    """
    logger.info("performing stratified sampling...")

    # separate fraud and normal
    fraud_df = df[df['isFraud'] == 1]
    normal_df = df[df['isFraud'] == 0]

    logger.info(f"total fraud cases: {len(fraud_df):,}")
    logger.info(f"total normal cases: {len(normal_df):,}")

    # keep ALL fraud cases — they're rare and precious
    # sample normal cases to reduce size
    normal_sampled = normal_df.sample(
        n=normal_sample_size,
        random_state=42
    )

    # combine fraud + sampled normal back together
    df_sampled = pd.concat([fraud_df, normal_sampled], ignore_index=True)

    # shuffle so fraud cases aren't all at the top
    df_sampled = df_sampled.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    logger.info(f"sampled dataset size: {len(df_sampled):,} rows")
    logger.info(f"fraud ratio after sampling: {df_sampled['isFraud'].mean():.2%}")

    return df_sampled


# -------------------------------------------------------------
# STEP 3 — Drop useless columns
# -------------------------------------------------------------
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that are useless or cause data leakage.

    Dropped columns:
    - nameOrig, nameDest: unique IDs, no ML signal
    - isFlaggedFraud: data leakage (bank's own flag)

    Args:
        df: dataframe after sampling

    Returns:
        dataframe with only useful columns
    """
    logger.info("dropping useless and leaky columns...")

    columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=columns_to_drop)

    logger.info(f"remaining columns: {df.columns.tolist()}")

    return df


# -------------------------------------------------------------
# STEP 4 — Encode categorical columns
# ML models only understand numbers, not text
# -------------------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts text columns to numbers for ML.

    'type' column: PAYMENT, TRANSFER etc → 0, 1, 2...
    This is called Label Encoding.

    Args:
        df: dataframe after dropping columns

    Returns:
        dataframe with encoded categorical columns
    """
    logger.info("encoding categorical columns...")

    le = LabelEncoder()

    # encode transaction type
    df['type'] = le.fit_transform(df['type'])

    # save the mapping so we can decode later in the UI
    type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"type encoding mapping: {type_mapping}")

    return df


# -------------------------------------------------------------
# STEP 5 — Add basic engineered features
# (advanced features come in Phase 5)
# -------------------------------------------------------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple but powerful derived features.

    These features capture patterns that raw columns miss:
    - balance difference tells us if account was drained
    - zero balance flags are strong fraud signals

    Args:
        df: encoded dataframe

    Returns:
        dataframe with new feature columns added
    """
    logger.info("adding basic engineered features...")

    # how much did sender balance change?
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']

    # how much did receiver balance change?
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # did sender account get completely drained?
    df['orig_balance_zero'] = (df['newbalanceOrig'] == 0).astype(int)

    # was receiver balance zero before receiving?
    df['dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)

    # transaction amount relative to sender's balance
    # avoid division by zero with np.where
    df['amount_to_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )

    logger.info("new features added: balance_diff_orig, balance_diff_dest, "
                "orig_balance_zero, dest_balance_zero, amount_to_balance_ratio")

    return df


# -------------------------------------------------------------
# STEP 6 — Save processed data
# -------------------------------------------------------------
def save_processed_data(df: pd.DataFrame,
                        output_path: str) -> None:
    """
    Saves the cleaned dataframe to CSV.

    Args:
        df: fully cleaned and processed dataframe
        output_path: where to save the file
    """
    logger.info(f"saving processed data to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"saved {len(df):,} rows successfully")


# -------------------------------------------------------------
# MASTER FUNCTION — runs entire pipeline in one call
# This is called the "orchestrator" pattern in industry
# -------------------------------------------------------------
def run_preprocessing_pipeline(
    input_path: str,
    output_path: str,
    normal_sample_size: int = 50000
) -> pd.DataFrame:
    """
    Runs the complete preprocessing pipeline end to end.

    This is the only function you need to call from outside.
    It orchestrates all steps in the correct order.

    Args:
        input_path: path to raw CSV
        output_path: path to save cleaned CSV
        normal_sample_size: how many normal rows to keep

    Returns:
        fully processed dataframe
    """
    logger.info("=" * 50)
    logger.info("starting preprocessing pipeline")
    logger.info("=" * 50)

    df = load_data(input_path)
    df = sample_data(df, normal_sample_size)
    df = drop_columns(df)
    df = encode_categorical(df)
    df = add_basic_features(df)
    save_processed_data(df, output_path)

    logger.info("=" * 50)
    logger.info("preprocessing pipeline complete!")
    logger.info(f"final shape: {df.shape}")
    logger.info("=" * 50)

    return df


# -------------------------------------------------------------
# This allows running the file directly from terminal
# python src/preprocessing.py
# -------------------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH = "data/raw/Transaction.csv"
    OUTPUT_PATH = "data/processed/transactions_clean.csv"

    df_clean = run_preprocessing_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        normal_sample_size=50000
    )

    print("\nfinal cleaned data sample:")
    print(df_clean.head())
    print(f"\nshape: {df_clean.shape}")
    print(f"\ncolumns: {df_clean.columns.tolist()}")