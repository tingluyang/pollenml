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
# Import necessary packages and functions
from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import seaborn as sns
import matplotlib as plt

# Some self defined functions
def clean_rare_labels(df, label_col, threshold=100):
    """
    Step 1: Remove categories with a sample size below the threshold.
    """
    counts = df[label_col].value_counts()
    to_keep = counts[counts >= threshold].index
    to_drop = counts[counts < threshold].index

    if len(to_drop) > 0:
        print(f"The following rare categories have been removed (sample size < {threshold}):")
        for label in to_drop:
            print(f"   - {label}: {counts[label]} samples")
    else:
        print("All categories meet the sample size requirements.")

    filtered_df = df[df[label_col].isin(to_keep)].copy()
    print(f"Remaining valid categories: {len(to_keep)}, Total sample size: {len(filtered_df)}")
    return filtered_df

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
    print(f" Total features required by model: {len(trained_features)}")
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

def export_proba_to_csv(predictor, data_pred, selected_models, save_dir):
    """
    Batch export prediction probabilities of multiple models to separate CSV files.

    Parameters:
        predictor: Trained TabularPredictor object
        data_pred: Data to be predicted (DataFrame or TabularDataset)
        selected_models: List of model names, e.g., ['LightGBM', 'XGBoost', 'RandomForest']
        save_dir: Directory path to save CSV files

    Returns:
        list: List of successfully saved file paths
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Store successfully saved file paths
    saved_files = []

    # Iterate through the selected model list
    for model_name in selected_models:
        try:
            # Perform probability prediction using the specified model
            proba_df = predictor.predict_proba(data_pred, model=model_name)

            # Ensure a DataFrame is returned (predict_proba usually returns a DataFrame)
            if not isinstance(proba_df, pd.DataFrame):
                proba_df = pd.DataFrame(proba_df)

            # Construct file path (use model name as filename)
            # Clean special characters in model name to avoid filename issues
            safe_model_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model_name)
            file_name = f"{safe_model_name}.csv"
            file_path = os.path.join(save_dir, file_name)

            # Add sample index (keep original data's index)
            proba_df.index = data_pred.index if hasattr(data_pred, 'index') else range(len(data_pred))

            # Save as CSV
            proba_df.to_csv(file_path)

            saved_files.append(file_path)
            print(f"✓ Model '{model_name}' prediction probabilities saved: {file_path} (shape: {proba_df.shape})")

        except Exception as e:
            print(f"✗ Model '{model_name}' processing failed: {str(e)}")
            continue

    print(f"\nComplete! Saved {len(saved_files)}/{len(selected_models)} files to: {save_dir}")

    return saved_files

def plot_feature_importance(importance_df, top_n=15):
    # 1. Formatting: convert index (feature name) to a column, and sort by importance
    fi_df = importance_df.reset_index()
    fi_df.columns = ['Feature', 'Importance', 'Std_Dev', 'P_Value', 'N']
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)

    # 2. Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=fi_df,
        x='Importance',
        y='Feature',
        palette='viridis'
    )

    # 3. Add error bars (optional, display std_dev)
    plt.errorbar(
        x=fi_df['Importance'],
        y=range(len(fi_df)),
        xerr=fi_df['Std_Dev'],
        fmt='none',
        c='black',
        capsize=3
    )

    plt.title(f'Top {top_n} Feature Importance (Permutation Importance)', fontsize=15)
    plt.xlabel('Performance Drop Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
