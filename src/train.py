# =============================================================
# src/train.py
# Production-Grade ML Training Pipeline
# AI Personal Finance Analyzer
# =============================================================
# WHAT THIS FILE DOES:
# - Loads feature engineered data
# - Splits with StratifiedKFold cross validation
# - Builds proper sklearn Pipelines
# - Tunes hyperparameters with RandomizedSearchCV
# - Evaluates all models professionally
# - Saves best models to disk
# =============================================================

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)
from xgboost import XGBClassifier
import joblib
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# STEP 1 — Load feature engineered data
# -------------------------------------------------------------
def load_features(filepath: str) -> tuple:
    """
    Loads feature engineered data and separates
    features (X) from target (y).

    Args:
        filepath: path to feature engineered CSV

    Returns:
        tuple of (X DataFrame, y Series)
    """
    logger.info(f"loading feature data from {filepath}")

    df = pd.read_csv(filepath)

    # drop columns not needed for training
    cols_to_drop = [c for c in ['isFraud', 'step', 'type']
                    if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df['isFraud']

    logger.info(f"X shape: {X.shape}")
    logger.info(f"fraud rate: {y.mean():.2%}")

    return X, y


# -------------------------------------------------------------
# STEP 2 — Stratified train/test split
# -------------------------------------------------------------
def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.2) -> tuple:
    """
    Splits data into train and test sets.

    Uses stratify=y to ensure both sets have
    the same fraud ratio — critical for imbalanced data.

    Args:
        X: features
        y: target
        test_size: fraction for test set

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"splitting data — test size: {test_size:.0%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    logger.info(f"train: {len(X_train):,} | test: {len(X_test):,}")
    logger.info(f"train fraud rate: {y_train.mean():.2%}")
    logger.info(f"test fraud rate:  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------
# STEP 3 — Cross validation helper
# evaluates model stability across multiple folds
# -------------------------------------------------------------
def cross_validate_model(pipeline: Pipeline,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         model_name: str,
                         cv_folds: int = 5) -> dict:
    """
    Runs StratifiedKFold cross validation on a pipeline.

    WHY CROSS VALIDATION:
    A single train/test split can be lucky or unlucky.
    Cross validation splits data into 5 folds, trains on 4,
    tests on 1 — repeated 5 times. Gives a stable,
    trustworthy performance estimate.

    StratifiedKFold ensures each fold has the same
    fraud ratio — essential for imbalanced datasets.

    Args:
        pipeline: sklearn Pipeline to evaluate
        X_train: training features
        y_train: training labels
        model_name: name for logging
        cv_folds: number of folds (default 5)

    Returns:
        dict with mean and std of CV scores
    """
    logger.info(f"running {cv_folds}-fold cross validation "
                f"for {model_name}...")

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    # score on ROC-AUC — best metric for fraud detection
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    logger.info(f"{model_name} CV AUC: "
                f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'cv_scores': cv_scores
    }


# -------------------------------------------------------------
# STEP 4 — Build and tune Logistic Regression Pipeline
# -------------------------------------------------------------
def build_logistic_regression(X_train: pd.DataFrame,
                               y_train: pd.Series) -> Pipeline:
    """
    Builds and tunes a Logistic Regression Pipeline.

    Pipeline = Scaler + Model in one object.
    This is the production-correct approach:
    scaler is fitted inside the pipeline so there
    is zero risk of data leakage during cross validation.

    Tunes: C (regularization strength)

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted Pipeline
    """
    logger.info("building Logistic Regression pipeline...")

    # define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # hyperparameter search space
    param_dist = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['lbfgs', 'saga'],
        'model__penalty': ['l2']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    logger.info("tuning Logistic Regression hyperparameters...")
    search.fit(X_train, y_train)

    logger.info(f"best params: {search.best_params_}")
    logger.info(f"best CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


# -------------------------------------------------------------
# STEP 5 — Build and tune Random Forest Pipeline
# -------------------------------------------------------------
def build_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series) -> Pipeline:
    """
    Builds and tunes a Random Forest Pipeline.

    Random Forest does NOT need scaling — trees split
    on thresholds, not distances. So our pipeline
    only has the model step.

    Tunes: n_estimators, max_depth, min_samples_split

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted Pipeline
    """
    logger.info("building Random Forest pipeline...")

    pipeline = Pipeline([
        ('model', RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # hyperparameter search space
    param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 15, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    logger.info("tuning Random Forest hyperparameters...")
    search.fit(X_train, y_train)

    logger.info(f"best params: {search.best_params_}")
    logger.info(f"best CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


# -------------------------------------------------------------
# STEP 6 — Build and tune XGBoost Pipeline
# -------------------------------------------------------------
def build_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series) -> Pipeline:
    """
    Builds and tunes an XGBoost Pipeline.

    XGBoost is gradient boosting — builds trees
    sequentially, each correcting previous mistakes.
    Industry standard for tabular fraud detection.

    scale_pos_weight handles class imbalance by
    giving more importance to fraud cases.

    Tunes: n_estimators, max_depth, learning_rate,
           subsample, colsample_bytree

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted Pipeline
    """
    logger.info("building XGBoost pipeline...")

    # calculate class weight for imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    pipeline = Pipeline([
        ('model', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    # hyperparameter search space
    param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 4, 5, 6, 7],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'model__min_child_weight': [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    logger.info("tuning XGBoost hyperparameters...")
    search.fit(X_train, y_train)

    logger.info(f"best params: {search.best_params_}")
    logger.info(f"best CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


# -------------------------------------------------------------
# STEP 7 — Evaluate a trained pipeline
# -------------------------------------------------------------
def evaluate_pipeline(pipeline: Pipeline,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str) -> dict:
    """
    Comprehensively evaluates a trained pipeline.

    Metrics explained:
    - ROC-AUC: overall discrimination ability (1.0 = perfect)
    - F1 Score: harmonic mean of precision and recall
    - Precision: of predicted fraud, how many are real fraud?
    - Recall: of all real fraud, how many did we catch?
    - PR-AUC: precision-recall AUC (better for imbalanced data)

    For fraud detection RECALL is most critical —
    missing real fraud costs more than false alarms.

    Args:
        pipeline: trained sklearn Pipeline
        X_test: test features
        y_test: true labels
        model_name: name for display

    Returns:
        dict of all evaluation metrics
    """
    logger.info(f"evaluating {model_name}...")

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # calculate all metrics
    roc_auc   = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    pr_auc    = average_precision_score(y_test, y_pred_prob)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred)

    # print professional summary
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  ROC-AUC Score    : {roc_auc:.4f}")
    print(f"  PR-AUC Score     : {pr_auc:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n  Classification Report:")
    print(report)

    return {
        'model_name': model_name,
        'roc_auc':    roc_auc,
        'pr_auc':     pr_auc,
        'f1_score':   f1,
        'precision':  precision,
        'recall':     recall,
        'confusion_matrix': cm,
        'pipeline':   pipeline
    }


# -------------------------------------------------------------
# STEP 8 — Save all pipelines and results
# -------------------------------------------------------------
def save_pipelines(results: dict,
                   output_dir: str) -> None:
    """
    Saves all trained pipelines to disk using joblib.

    Each pipeline contains the full preprocessing +
    model in one object — ready for production inference.

    Also saves the best model separately for easy loading.

    Args:
        results: dict of evaluation results
        output_dir: folder to save models
    """
    logger.info(f"saving pipelines to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # save each pipeline
    for name, result in results.items():
        filepath = os.path.join(output_dir, f"{name}_pipeline.pkl")
        joblib.dump(result['pipeline'], filepath)
        logger.info(f"saved: {filepath}")

    # find and save best model separately
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_pipeline = results[best_name]['pipeline']
    best_path = os.path.join(output_dir, "best_model_pipeline.pkl")
    joblib.dump(best_pipeline, best_path)

    logger.info(f"best model: {best_name} "
                f"(AUC: {results[best_name]['roc_auc']:.4f})")
    logger.info(f"saved best model: {best_path}")


# -------------------------------------------------------------
# MASTER FUNCTION
# -------------------------------------------------------------
def run_training_pipeline(input_path: str,
                          models_dir: str) -> dict:
    """
    Runs the complete production-grade training pipeline.

    Flow:
    1. Load features
    2. Stratified train/test split
    3. Build + tune 3 pipelines with RandomizedSearchCV
    4. Cross validate each pipeline
    5. Evaluate on held-out test set
    6. Save all pipelines

    Args:
        input_path: path to feature engineered CSV
        models_dir: folder to save trained pipelines

    Returns:
        dict of evaluation results for all models
    """
    logger.info("=" * 55)
    logger.info("starting production training pipeline")
    logger.info("=" * 55)

    # load and split data
    X, y = load_features(input_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── build and tune all 3 pipelines ──────────────────────
    lr_pipeline  = build_logistic_regression(X_train, y_train)
    rf_pipeline  = build_random_forest(X_train, y_train)
    xgb_pipeline = build_xgboost(X_train, y_train)

    # ── cross validate all pipelines ────────────────────────
    print("\n" + "="*55)
    print("  CROSS VALIDATION RESULTS")
    print("="*55)

    for name, pipe in [
        ("Logistic Regression", lr_pipeline),
        ("Random Forest",       rf_pipeline),
        ("XGBoost",             xgb_pipeline)
    ]:
        cv_result = cross_validate_model(pipe, X_train, y_train, name)
        print(f"  {name:<25} "
              f"AUC: {cv_result['cv_mean']:.4f} "
              f"± {cv_result['cv_std']:.4f}")

    # ── evaluate on test set ─────────────────────────────────
    results = {}

    results['logistic_regression'] = evaluate_pipeline(
        lr_pipeline, X_test, y_test, "Logistic Regression"
    )
    results['random_forest'] = evaluate_pipeline(
        rf_pipeline, X_test, y_test, "Random Forest"
    )
    results['xgboost'] = evaluate_pipeline(
        xgb_pipeline, X_test, y_test, "XGBoost"
    )

    # ── save all pipelines ───────────────────────────────────
    save_pipelines(results, models_dir)

    # ── final leaderboard ────────────────────────────────────
    print(f"\n{'='*55}")
    print("  FINAL MODEL LEADERBOARD")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'AUC':>8} {'F1':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for name, result in sorted(results.items(),
                                key=lambda x: x[1]['roc_auc'],
                                reverse=True):
        print(f"  {result['model_name']:<25} "
              f"{result['roc_auc']:>8.4f} "
              f"{result['f1_score']:>8.4f} "
              f"{result['precision']:>10.4f} "
              f"{result['recall']:>8.4f}")

    logger.info("training pipeline complete!")

    return results


# -------------------------------------------------------------
# Run directly from terminal
# python src/train.py
# -------------------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH = "data/processed/transactions_features.csv"
    MODELS_DIR = "models/"

    results = run_training_pipeline(
        input_path=INPUT_PATH,
        models_dir=MODELS_DIR
    )