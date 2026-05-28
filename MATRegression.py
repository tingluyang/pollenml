"""
=================== System Info ===================
AutoGluon Version:  1.5.0
Python Version:     3.12.13
CPU Count:          12
Pytorch Version:    2.9.1+cu128
CUDA Version:       12.8
GPU Memory:         GPU 0: 22.03/22.03 GB
Total GPU Memory:   Free: 22.03 GB, Allocated: 0.00 GB, Total: 22.03 GB
GPU Count:          1
Memory Avail:       48.76 GB / 52.96 GB (92.1%)
===================================================
"""

# This is NOT an automated script. It is recommended to run it in Jupyter.

# Import necessary packages and functions
from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List, Optional
import shap
import time

def process_pollen_pipeline(df, label_col, test_size=0.2, min_samples=2, min_percent=1.0):
    """
    Adaptive pollen data processing pipeline for task types (classification/regression).
    Returns: train_final, test_final, dropped_features
    """
    df_work = df.copy()

    # --- 1. Auto-detect task type ---
    target_series = df_work[label_col]
    is_regression = False
    if np.issubdtype(target_series.dtype, np.floating) or (np.issubdtype(target_series.dtype, np.integer) and target_series.nunique() > 15):
        is_regression = True

    # --- 2. Prepare stratification strategy ---
    stratify_col = pd.qcut(target_series, q=5, labels=False, duplicates='drop') if is_regression else target_series

    # --- 3. Split dataset ---
    train_df, test_df = train_test_split(
        df_work,
        test_size=test_size,
        stratify=stratify_col,
        random_state=42
    )

    # --- 4. Feature selection (based only on training set criteria) ---
    X_train_raw = train_df.drop(columns=[label_col])
    # Core criterion: Content > min_percent in at least min_samples
    mask = (X_train_raw > min_percent).sum(axis=0) >= min_samples

    selected_features = X_train_raw.columns[mask].tolist()
    dropped_features = X_train_raw.columns[~mask].tolist()

    # --- Print display area ---
    print(f"\n" + "="*50)
    print(f"Feature Selection Report (Task: {'Regression' if is_regression else 'Classification'})")
    print(f"Criterion: Percentage content > {min_percent}% in at least {min_samples} samples")
    print(f"Retained features: {len(selected_features)} | Dropped features: {len(dropped_features)}")
    print("-" * 50)

    if len(dropped_features) > 0:
        print(f"The following {len(dropped_features)} features have been dropped:")
        # Print all names with line breaks, 5 per line for easy reading
        for i in range(0, len(dropped_features), 5):
            print("  " + ", ".join(dropped_features[i:i+5]))
    else:
        print("No features were dropped.")
    print("="*50 + "\n")

    # --- 5. Hellinger Transformation ---
    def apply_hellinger(data_df, features, target_col):
        """
        Modified Hellinger transformation:
        Automatically handles the 're-closure' problem after feature removal.
        """
        # 1. Extract selected feature columns
        X = data_df[features].copy()

        # 2. Calculate current sum of each row (Row Sum)
        # This ensures the remaining parts are re-normalized even if rare features are removed
        row_sums = X.sum(axis=1)

        # 3. Perform Hellinger transformation: sqrt( x_ij / row_sum_i )
        # Use div to ensure row-aligned division
        X_transformed = np.sqrt(X.div(row_sums, axis=0))

        # 4. Concatenate labels
        return pd.concat([X_transformed, data_df[target_col].reset_index(drop=True)], axis=1)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_final = apply_hellinger(train_df, selected_features, label_col)
    test_final = apply_hellinger(test_df, selected_features, label_col)

    return train_final, test_final

# Inference dataset processing
def prepare_inference_data(file_path, predictor_models):

    trained_features = predictor_models.feature_metadata.get_features()
    """
    Feature transposition, alignment, and difference reporting function for prediction tasks.
    """
    # 1. Read CSV, use the first column as index (feature names)
    # Allow null values, fill with 0 uniformly later
    df_raw = pd.read_csv(file_path, index_col=0).fillna(0)

    # 2. Transpose data: samples become rows, features become columns
    df_T = df_raw.T
    input_features = df_T.columns.tolist()

    # --- 3. Feature comparison and print report ---
    # A. Unmatched features (present in prediction table, but not in training set -> discard)
    unmatched_features = [f for f in input_features if f not in trained_features]

    # B. Features present in training set but missing in prediction table (auto-fill with 0)
    missing_in_input = [f for f in trained_features if f not in input_features]

    print("\n" + "="*60)
    print("Inference Data Feature Matching Report")
    print("-" * 60)
    print(f"Total features required by model: {len(trained_features)}")
    print(f"Total features provided by inference table: {len(input_features)}")

    if len(unmatched_features) > 0:
        print(f"Found {len(unmatched_features)} features that cannot be matched with the model (will be ignored):")
        # Print 5 per line for easy reading
        for i in range(0, len(unmatched_features), 5):
            print(f"   - {', '.join(unmatched_features[i:i+5])}")
    else:
        print("All features in the inference table are covered by the model.")

    if len(missing_in_input) > 0:
        print(f"{len(missing_in_input)} features from the training set are missing in the inference table (auto-filled with 0)")
    print("-" * 60)

    # 4. Feature alignment
    # Construct a new DataFrame strictly following the feature order from training
    X_inference = pd.DataFrame(0.0, index=df_T.index, columns=trained_features)

    # Copy only the features required by the model
    matched_cols = [f for f in trained_features if f in input_features]
    X_inference[matched_cols] = df_T[matched_cols]

    # 5. Re-closure processing
    # Extremely important: recalculate row sums based on remaining features
    row_sums = X_inference.sum(axis=1)

    # Handle all-zero rows to prevent division by zero errors
    if (row_sums == 0).any():
        zero_samples = row_sums[row_sums == 0].index.tolist()
        print(f"CRITICAL WARNING: Samples {zero_samples} do not contain any features recognized by the model! Prediction results may be unreliable.")
        row_sums = row_sums.replace(0, 1)

    # 6. Hellinger transformation: sqrt( x / row_sum )
    X_final = np.sqrt(X_inference.div(row_sums, axis=0))

    print(f"Final output format: {X_final.shape[0]} samples x {X_final.shape[1]} features")
    print("="*60 + "\n")

    return X_final

def multi_model_predict(predictor, data_pred, selected_models):
    """
    Predict using multiple AutoGluon models and integrate the results.

    Parameters:
        predictor: Trained TabularPredictor object
        data_pred: Data to be predicted (DataFrame or TabularDataset)
        selected_models: List of model names, e.g., ['LightGBM', 'XGBoost', 'RandomForest']

    Returns:
        DataFrame: Contains prediction results for each model, with each column corresponding to a model.
    """

    # Store prediction results for each model
    predictions_dict = {}

    # Iterate through the selected model list
    for model_name in selected_models:
        try:
            # Predict using the specified model
            pred = predictor.predict(data_pred, model=model_name)

            # Store prediction results in dictionary, using model name as column name
            predictions_dict[model_name] = pred

            print(f"✓ Model '{model_name}' prediction complete, number of samples: {len(pred)}")

        except Exception as e:
            print(f"✗ Model '{model_name}' prediction failed: {str(e)}")
            # Optional: Skip failed models or fill with NaN
            predictions_dict[model_name] = [None] * len(data_pred)

    # Integrate all prediction results into a DataFrame
    result_df = pd.DataFrame(predictions_dict)

    # Add sample index (keep original data's index)
    result_df.index = data_pred.index if hasattr(data_pred, 'index') else range(len(data_pred))

    print(f"\nPrediction complete! Result shape: {result_df.shape}")

    return result_df


def export_and_plot_feature_importance(predictor, model_sel, test_data, n=15, save_dir='importance_results'):
    """
    Calculate feature importance for a specified list of AutoGluon models, export to CSV and save visualizations as PDF.
    """
    # 1. Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Get all available model names in the current predictor
    available_models = predictor.model_names()

    for model_name in model_sel:
        if model_name not in available_models:
            print(f"Skipping model [{model_name}]: Not found in Predictor.")
            continue

        print(f"Processing model: {model_name} ...")

        try:
            # 2. Calculate feature importance (Permutation Importance)
            # If data volume is huge, it is recommended to add the subsample_size=1000 parameter here
            fi_df = predictor.feature_importance(test_data, model=model_name, subsample_size =200)

            # 3. Export complete feature importance to CSV
            # Clean the model name to prevent errors when used as a filename (e.g., remove slashes)
            safe_model_name = model_name.replace('/', '_').replace('\\', '_')
            csv_file = save_path / f"{safe_model_name}.csv"
            fi_df.to_csv(csv_file)

            # 4. Prepare plotting data (take top n)
            plot_data = fi_df.reset_index().rename(columns={'index': 'Feature'})
            plot_data = plot_data.sort_values(by='importance', ascending=False).head(n)

            # 5. Start plotting
            plt.figure(figsize=(10, 8))
            sns.set_style("whitegrid")

            # Display using bar chart
            barplot = sns.barplot(
                data=plot_data,
                x='importance',
                y='Feature',
                palette='viridis'
            )

            # Add numerical annotations
            for p in barplot.patches:
                barplot.annotate(f"{p.get_width():.4f}",
                                 (p.get_width(), p.get_y() + p.get_height() / 2),
                                 ha='left', va='center', fontsize=9, color='black', xytext=(5, 0),
                                 textcoords='offset points')

            plt.title(f'Top {n} Feature Importance\nModel: {model_name}', fontsize=14)
            plt.xlabel('Importance Score (Drop in Performance)', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()

            # 6. Save as PDF
            pdf_file = save_path / f"{safe_model_name}.pdf"
            plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
            plt.close() # Free memory

            print(f"  [Complete] Results saved to: {save_dir}/{safe_model_name}.csv/pdf")

        except Exception as e:
            print(f"  [Error] Exception occurred while calculating model {model_name}: {e}")

def evaluate_and_save_model(
    model,
    model_name: str,
    test_data=None,
    output_dir: str = "/content/drive/MyDrive/Scientific/PollenML",
    extra_metrics: Optional[List[str]] = None
) -> dict:
    """
    Evaluate AutoGluon model performance and save results to CSV files.

    Parameters:
    -----------
    model : TabularPredictor
        Trained AutoGluon model
    model_name : str
        Model identifier name (e.g., 'v4_best', 'v4_extreme')
    test_data : TabularDataset, optional
        Test dataset; if provided, evaluate test set performance
    output_dir : str
        Output directory path
    extra_metrics : list, optional
        List of extra evaluation metrics for the test set

    Returns:
    --------
    dict : Dictionary containing training and testing (if any) leaderboards
    """

    # Default extra evaluation metrics
    default_extra_metrics = [
        'accuracy', 'acc', 'balanced_accuracy', 'mcc',
        'log_loss', 'nll', 'pac', 'pac_score', 'quadratic_kappa',
        'precision_macro', 'precision_micro', 'precision_weighted',
        'recall_macro', 'recall_micro', 'recall_weighted',
        'f1_macro', 'f1_micro', 'f1_weighted',
        'roc_auc_ovo', 'roc_auc_ovo_macro', 'roc_auc_ovo_weighted',
        'roc_auc_ovr', 'roc_auc_ovr_macro', 'roc_auc_ovr_micro', 'roc_auc_ovr_weighted'
    ]

    # Use provided metrics or default metrics
    metrics = extra_metrics if extra_metrics is not None else default_extra_metrics

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # ========== Training Set Evaluation ==========
    print(f"\n{'='*50}")
    print(f"Training Leaderboard - {model_name}")
    print(f"{'='*50}")

    train_lb = model.leaderboard()
    print(train_lb)

    # Save training set results
    train_path = os.path.join(output_dir, f"{model_name}_train_lb.csv")
    train_lb.to_csv(train_path, index=False)
    print(f"✓ Saved to: {train_path}")

    results['train'] = train_lb

    # ========== Testing Set Evaluation (if test data is provided) ==========
    if test_data is not None:
        print(f"\n{'='*50}")
        print(f"Testing Leaderboard - {model_name}")
        print(f"{'='*50}")

        test_lb = model.leaderboard(test_data, extra_metrics=metrics)
        print(test_lb)

        # Save testing set results
        test_path = os.path.join(output_dir, f"{model_name}_test_lb.csv")
        test_lb.to_csv(test_path, index=False)
        print(f"✓ Saved to: {test_path}")

        results['test'] = test_lb

    return results
