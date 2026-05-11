# =============================================================
# src/evaluate.py
# Professional Model Evaluation & Visualization
# AI Personal Finance Analyzer
# =============================================================
# WHAT THIS FILE DOES:
# - Loads all trained pipelines
# - Generates ROC curves for all models
# - Generates Precision-Recall curves
# - Generates confusion matrix heatmaps
# - Generates feature importance chart
# - Saves all charts to reports/ folder
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
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
# Global dark theme for all charts
# -------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor':  '#0e1117',
    'axes.facecolor':    '#0e1117',
    'axes.edgecolor':    '#2d2d2d',
    'axes.labelcolor':   '#e0e0e0',
    'text.color':        '#e0e0e0',
    'xtick.color':       '#a0a0a0',
    'ytick.color':       '#a0a0a0',
    'grid.color':        '#2d2d2d',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.facecolor':  '#1a1a2e',
    'legend.edgecolor':  '#2d2d2d',
})

# color palette
COLORS = {
    'logistic_regression': '#7c6fe0',
    'random_forest':       '#00d4aa',
    'xgboost':             '#ff4b4b',
    'diagonal':            '#555555',
    'fraud':               '#ff4b4b',
    'normal':              '#00d4aa',
}


# -------------------------------------------------------------
# STEP 1 — Load data and models
# -------------------------------------------------------------
def load_evaluation_data(data_path: str,
                         models_dir: str) -> tuple:
    """
    Loads test data and all trained pipelines for evaluation.

    Args:
        data_path: path to feature engineered CSV
        models_dir: folder containing saved pipelines

    Returns:
        tuple of (X_test, y_test, models_dict)
    """
    logger.info("loading evaluation data and models...")

    # load feature data
    df = pd.read_csv(data_path)

    cols_to_drop = [c for c in ['isFraud', 'step', 'type']
                    if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df['isFraud']

    # use same split as training for consistency
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # load all pipelines
    models = {}
    model_files = {
        'logistic_regression': 'logistic_regression_pipeline.pkl',
        'random_forest':       'random_forest_pipeline.pkl',
        'xgboost':             'xgboost_pipeline.pkl'
    }

    for name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        models[name] = joblib.load(filepath)
        logger.info(f"loaded: {filename}")

    logger.info(f"test set size: {len(X_test):,}")
    logger.info(f"fraud in test: {y_test.sum():,}")

    return X_test, y_test, models


# -------------------------------------------------------------
# STEP 2 — ROC Curves for all models
# -------------------------------------------------------------
def plot_roc_curves(models: dict,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    output_path: str) -> None:
    """
    Plots ROC curves for all models on one chart.

    ROC curve shows tradeoff between:
    - True Positive Rate (catching real fraud)
    - False Positive Rate (false alarms)

    AUC = Area Under Curve. 1.0 = perfect, 0.5 = random.

    Args:
        models: dict of trained pipelines
        X_test: test features
        y_test: true labels
        output_path: where to save the chart
    """
    logger.info("plotting ROC curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest':       'Random Forest',
        'xgboost':             'XGBoost'
    }

    for name, pipeline in models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            color=COLORS[name],
            linewidth=2.5,
            label=f"{model_display_names[name]} (AUC = {roc_auc:.4f})"
        )

    # diagonal random classifier line
    ax.plot(
        [0, 1], [0, 1],
        color=COLORS['diagonal'],
        linewidth=1.5,
        linestyle='--',
        label='Random Classifier (AUC = 0.5000)'
    )

    # fill area under best curve
    best_name = max(models.keys(),
                    key=lambda k: roc_auc_score(
                        y_test,
                        models[k].predict_proba(X_test)[:, 1]
                    ))
    y_prob_best = models[best_name].predict_proba(X_test)[:, 1]
    fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_best)
    ax.fill_between(fpr_best, tpr_best, alpha=0.08,
                    color=COLORS[best_name])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    logger.info(f"saved: {output_path}")


# -------------------------------------------------------------
# STEP 3 — Precision-Recall Curves
# -------------------------------------------------------------
def plot_precision_recall_curves(models: dict,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series,
                                  output_path: str) -> None:
    """
    Plots Precision-Recall curves for all models.

    PR curves are MORE informative than ROC for
    imbalanced datasets like fraud detection.

    High precision = few false alarms
    High recall = catch most real fraud

    Args:
        models: dict of trained pipelines
        X_test: test features
        y_test: true labels
        output_path: where to save the chart
    """
    logger.info("plotting precision-recall curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest':       'Random Forest',
        'xgboost':             'XGBoost'
    }

    for name, pipeline in models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(
            y_test, y_prob
        )
        pr_auc = average_precision_score(y_test, y_prob)

        ax.plot(
            recall, precision,
            color=COLORS[name],
            linewidth=2.5,
            label=f"{model_display_names[name]} (PR-AUC = {pr_auc:.4f})"
        )

    # baseline — random classifier
    baseline = y_test.sum() / len(y_test)
    ax.axhline(
        y=baseline,
        color=COLORS['diagonal'],
        linewidth=1.5,
        linestyle='--',
        label=f'Random Classifier (PR-AUC = {baseline:.4f})'
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — Model Comparison',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    logger.info(f"saved: {output_path}")


# -------------------------------------------------------------
# STEP 4 — Confusion Matrix Heatmaps
# -------------------------------------------------------------
def plot_confusion_matrices(models: dict,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             output_path: str) -> None:
    """
    Plots confusion matrices for all 3 models side by side.

    Confusion matrix shows:
    - TN (True Negatives):  normal correctly identified
    - FP (False Positives): normal wrongly flagged as fraud
    - FN (False Negatives): fraud missed — most costly!
    - TP (True Positives):  fraud correctly caught

    Args:
        models: dict of trained pipelines
        X_test: test features
        y_test: true labels
        output_path: where to save the chart
    """
    logger.info("plotting confusion matrices...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confusion Matrices — All Models',
                 fontsize=16, fontweight='bold', y=1.02)

    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest':       'Random Forest',
        'xgboost':             'XGBoost'
    }

    for idx, (name, pipeline) in enumerate(models.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # normalize to percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # create annotation with both count and percentage
        annotations = np.array([
            [f'{cm[i,j]:,}\n({cm_normalized[i,j]:.1%})'
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        sns.heatmap(
            cm_normalized,
            annot=annotations,
            fmt='',
            cmap='RdYlGn',
            ax=axes[idx],
            vmin=0, vmax=1,
            linewidths=2,
            linecolor='#0e1117',
            annot_kws={'size': 12, 'weight': 'bold'}
        )

        axes[idx].set_title(
            model_display_names[name],
            fontsize=13, fontweight='bold', pad=12
        )
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xticklabels(
            ['Normal', 'Fraud'], fontsize=10
        )
        axes[idx].set_yticklabels(
            ['Normal', 'Fraud'], fontsize=10, rotation=0
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    logger.info(f"saved: {output_path}")


# -------------------------------------------------------------
# STEP 5 — Feature Importance Chart
# -------------------------------------------------------------
def plot_feature_importance(models: dict,
                             feature_names: list,
                             output_path: str) -> None:
    """
    Plots feature importance from Random Forest and XGBoost.

    Feature importance tells us which features the model
    relies on most for predictions. This is crucial for:
    - Model interpretability
    - Business insights
    - Debugging bad predictions

    Args:
        models: dict of trained pipelines
        feature_names: list of feature column names
        output_path: where to save the chart
    """
    logger.info("plotting feature importance...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Feature Importance Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    tree_models = {
        'random_forest': ('Random Forest', COLORS['random_forest']),
        'xgboost':       ('XGBoost',       COLORS['xgboost'])
    }

    for idx, (name, (display_name, color)) in enumerate(
        tree_models.items()
    ):
        pipeline = models[name]

        # extract model from pipeline
        model = pipeline.named_steps['model']
        importances = model.feature_importances_

        # create dataframe and sort
        feat_df = pd.DataFrame({
            'feature':    feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)

        # plot top 15 features
        top_n = 15
        feat_df = feat_df.tail(top_n)

        bars = axes[idx].barh(
            feat_df['feature'],
            feat_df['importance'],
            color=color,
            edgecolor='none',
            height=0.7,
            alpha=0.85
        )

        # add value labels
        for bar, val in zip(bars, feat_df['importance']):
            axes[idx].text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}',
                va='center', ha='left',
                color='#a0a0a0', fontsize=9
            )

        axes[idx].set_title(
            f'{display_name} — Top {top_n} Features',
            fontsize=13, fontweight='bold', pad=12
        )
        axes[idx].set_xlabel('Importance Score', fontsize=11)
        axes[idx].xaxis.grid(True)
        axes[idx].set_axisbelow(True)
        axes[idx].spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    logger.info(f"saved: {output_path}")


# -------------------------------------------------------------
# STEP 6 — Model Comparison Dashboard
# -------------------------------------------------------------
def plot_model_comparison(models: dict,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           output_path: str) -> None:
    """
    Creates a summary dashboard comparing all models
    across all key metrics.

    This is the single most impressive chart for your
    README — shows everything at a glance.

    Args:
        models: dict of trained pipelines
        X_test: test features
        y_test: true labels
        output_path: where to save the chart
    """
    logger.info("plotting model comparison dashboard...")

    model_display_names = {
        'logistic_regression': 'Logistic\nRegression',
        'random_forest':       'Random\nForest',
        'xgboost':             'XGBoost'
    }

    metrics = ['ROC-AUC', 'PR-AUC', 'F1 Score',
               'Precision', 'Recall']
    results = {name: {} for name in models}

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        results[name]['ROC-AUC']   = roc_auc_score(y_test, y_prob)
        results[name]['PR-AUC']    = average_precision_score(
            y_test, y_prob
        )
        results[name]['F1 Score']  = f1_score(y_test, y_pred)
        results[name]['Precision'] = precision_score(y_test, y_pred)
        results[name]['Recall']    = recall_score(y_test, y_pred)

    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(20, 7))
    fig.suptitle('Model Performance Comparison Dashboard',
                 fontsize=16, fontweight='bold', y=1.02)

    model_names = list(models.keys())
    colors = [COLORS[name] for name in model_names]
    x_labels = [model_display_names[n] for n in model_names]

    for idx, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]

        bars = axes[idx].bar(
            x_labels, values,
            color=colors,
            edgecolor='none',
            width=0.5
        )

        # add value labels on top
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{val:.4f}',
                ha='center', va='bottom',
                color='white', fontsize=9,
                fontweight='bold'
            )

        axes[idx].set_title(metric, fontsize=12,
                             fontweight='bold', pad=10)
        axes[idx].set_ylim([0.88, 1.01])
        axes[idx].yaxis.grid(True)
        axes[idx].set_axisbelow(True)
        axes[idx].tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    logger.info(f"saved: {output_path}")


# -------------------------------------------------------------
# MASTER FUNCTION
# -------------------------------------------------------------
def run_evaluation_pipeline(data_path: str,
                             models_dir: str,
                             reports_dir: str) -> None:
    """
    Runs the complete evaluation and visualization pipeline.

    Generates 5 professional charts saved to reports/:
    1. ROC curves
    2. Precision-Recall curves
    3. Confusion matrices
    4. Feature importance
    5. Model comparison dashboard

    Args:
        data_path: path to feature engineered CSV
        models_dir: folder with saved pipelines
        reports_dir: folder to save charts
    """
    logger.info("=" * 55)
    logger.info("starting evaluation pipeline")
    logger.info("=" * 55)

    os.makedirs(reports_dir, exist_ok=True)

    # load data and models
    X_test, y_test, models = load_evaluation_data(
        data_path, models_dir
    )

    # get feature names
    feature_names = list(X_test.columns)

    # generate all charts
    plot_roc_curves(
        models, X_test, y_test,
        os.path.join(reports_dir, '06_roc_curves.png')
    )

    plot_precision_recall_curves(
        models, X_test, y_test,
        os.path.join(reports_dir, '07_precision_recall_curves.png')
    )

    plot_confusion_matrices(
        models, X_test, y_test,
        os.path.join(reports_dir, '08_confusion_matrices.png')
    )

    plot_feature_importance(
        models, feature_names,
        os.path.join(reports_dir, '09_feature_importance.png')
    )

    plot_model_comparison(
        models, X_test, y_test,
        os.path.join(reports_dir, '10_model_comparison.png')
    )

    logger.info("=" * 55)
    logger.info("evaluation pipeline complete!")
    logger.info(f"5 charts saved to {reports_dir}")
    logger.info("=" * 55)


# -------------------------------------------------------------
# Run directly from terminal
# python src/evaluate.py
# -------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH   = "data/processed/transactions_features.csv"
    MODELS_DIR  = "models/"
    REPORTS_DIR = "reports/"

    run_evaluation_pipeline(
        data_path=DATA_PATH,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR
    )