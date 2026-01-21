# add evaluation regarding baselines (random, HBI, PLA2)
# add evaluation regarding low confidences + FPs


import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score
from sklearn.dummy import DummyClassifier


# -----------------------------------------------------------------------------
# 1. BASELINE EVALUATION
# -----------------------------------------------------------------------------
def evaluate_baselines(y_true, hbi_preds=None, pla2_preds=None):
    """
    Evaluates Random, HBI, and PLA2 baselines against the Ground Truth.
    """
    print("\n" + "=" * 40)
    print("BASELINE EVALUATION")
    print("=" * 40)

    # --- A. Random Baseline (Stratified) ---
    # Simulates a model that guesses based on class distribution
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(np.zeros((len(y_true), 1)), y_true)  # X doesn't matter for dummy
    random_preds = dummy.predict(np.zeros((len(y_true), 1)))

    print(f"\n[Random Baseline] MCC: {matthews_corrcoef(y_true, random_preds):.3f}")
    print(classification_report(y_true, random_preds, target_names=["Non-Tox", "Tox"], zero_division=0))

    # --- B. HBI Baseline (Homology Based Inference) ---
    if hbi_preds is not None:
        # Assuming hbi_preds is a Series/List aligned with y_true
        # Handle 'no hit' or NaNs by treating them as the negative class (0) or a separate category
        hbi_clean = hbi_preds.fillna(0).astype(int)

        print(f"\n[HBI Baseline] MCC: {matthews_corrcoef(y_true, hbi_clean):.3f}")
        print(classification_report(y_true, hbi_clean, target_names=["Non-Tox", "Tox"], zero_division=0))
    else:
        print("\n[HBI Baseline] Skipped (No data provided)")

    # --- C. PLA2 Baseline ---
    if pla2_preds is not None:
        # Placeholder for PLA2 specific logic
        print(f"\n[PLA2 Baseline] MCC: {matthews_corrcoef(y_true, pla2_preds):.3f}")
        print(classification_report(y_true, pla2_preds, target_names=["Non-Tox", "Tox"], zero_division=0))
    else:
        print("\n[PLA2 Baseline] Skipped (No data provided)")


# -----------------------------------------------------------------------------
# 2. LOW CONFIDENCE & FALSE POSITIVE ANALYSIS
# -----------------------------------------------------------------------------
def analyze_model_errors(df, prob_col='probability', label_col='label', pred_col='prediction'):
    """
    Analyzes where the model is uncertain and where it makes False Positive errors.
    Expects a DataFrame containing model predictions and ground truth.
    """
    print("\n" + "=" * 40)
    print("ERROR & CONFIDENCE ANALYSIS")
    print("=" * 40)

    # --- A. Confidence Distribution ---
    # Create bins for confidence to see how accuracy degrades with uncertainty
    # We assume probability is P(Class 1). Confidence is distance from 0.5
    df['confidence'] = (df[prob_col] - 0.5).abs() * 2  # Scales 0.5-1.0 to 0.0-1.0

    # Define Low Confidence as < 0.7 (adjust threshold as needed)
    low_conf_threshold = 0.4  # approx probability 0.3 to 0.7
    low_conf_df = df[df['confidence'] < low_conf_threshold]

    print(f"\n--- Low Confidence Samples (Conf < {low_conf_threshold}) ---")
    print(f"Count: {len(low_conf_df)} / {len(df)} ({len(low_conf_df) / len(df):.1%})")
    if len(low_conf_df) > 0:
        print(f"Accuracy on Low Conf: {accuracy_score(low_conf_df[label_col], low_conf_df[pred_col]):.3f}")
        # Show a few examples
        print("Sample Low Conf predictions:")
        print(low_conf_df[[label_col, pred_col, prob_col]].head())

    # --- B. False Positive Analysis ---
    # FP: True=0, Pred=1
    fps = df[(df[label_col] == 0) & (df[pred_col] == 1)].copy()

    print(f"\n--- False Positives (FPs) ---")
    print(f"Total FPs: {len(fps)}")

    if len(fps) > 0:
        avg_fp_conf = fps[prob_col].mean()
        print(f"Average Model Confidence on FPs: {avg_fp_conf:.3f}")

        # Are FPs mostly low confidence?
        high_conf_fps = fps[fps[prob_col] > 0.9]
        print(f"High Confidence FPs (>0.9): {len(high_conf_fps)}")

        if not high_conf_fps.empty:
            print("Top High-Confidence FPs (Potential Label Errors?):")
            # Assuming there is an 'id' column, otherwise index
            cols_to_show = ['id', prob_col] if 'id' in df.columns else [prob_col]
            print(high_conf_fps[cols_to_show].head())


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load your model predictions (Output from training)
    # Ensure this CSV has: 'label' (True), 'prediction' (Pred), 'probability' (Score)
    pred_file = "model/model_output/test_predictions.csv"

    try:
        df = pd.read_csv(pred_file)

        # 2. Run Baselines
        # For HBI/PLA2, you would load them here.
        # Passing None skips them for now as requested.
        evaluate_baselines(y_true=df['label'], hbi_preds=None, pla2_preds=None)

        # 3. Run Error Analysis
        analyze_model_errors(df)

    except FileNotFoundError:
        print(f"Error: Could not find {pred_file}. Please run training first.")