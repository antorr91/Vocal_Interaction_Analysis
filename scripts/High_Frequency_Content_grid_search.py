import os
import glob
import json
from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from  src.detection import onset_detection_algorithms as onset_detectors
from src.detection import evaluation as my_eval
from mir_eval_modified.onset import f_measure

# =========================
# CONFIG
# =========================
EVAL_WINDOW = 0.05  # default eval window (overridden only in 'evaluation' search)

audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\Normalised_Audio\Data_grid_search'
metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\metadata_testing_1.csv'
output_directory = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\output_grid_search'

os.makedirs(output_directory, exist_ok=True)

# Which search to run:
# 'input_features', 'peak_picking', 'evaluation', 'full_search'
which_search = 'full_search'

# Output files
if which_search == 'input_features':
    output_file = os.path.join(output_directory, "HFC_search_input_features.json")
elif which_search == 'peak_picking':
    output_file = os.path.join(output_directory, "HFC_search_peak_picking.json")
elif which_search == 'evaluation':
    output_file = os.path.join(output_directory, "HFC_search_evaluation.json")
elif which_search == 'full_search':
    output_file = os.path.join(output_directory, "HFC_full_grid_search.json")
else:
    raise ValueError(f"Invalid which_search: {which_search}")

csv_summary = output_file.replace(".json", "_grid_results_summary.csv")

# Fixed HFC frontend parameters
hop_length = 441
sr = 44100

# =========================
# LOAD METADATA (same structure as your eval script)
# =========================
metadata = pd.read_csv(metadata_path)
for col in ['Filename', 'Start_experiment_sec', 'End_experiment_sec']:
    if col not in metadata.columns:
        raise ValueError(f"Missing column '{col}' in metadata CSV.")
metadata['Start_experiment_sec'] = pd.to_numeric(metadata['Start_experiment_sec'], errors='coerce')
metadata['End_experiment_sec'] = pd.to_numeric(metadata['End_experiment_sec'], errors='coerce')

# =========================
# PARAMETER RANGES (YOUR combos)
# =========================
# Input features
num_bands_range = [24,30, 32]
fmin_range      = [1000, 1500]
fmax_range      = [10000, 14000]  # must be < Nyquist (22050 at sr=44100)
fref_range      = [2400, 2800, 3000, 3200]
# Constraint enforced below: fmin < fref < fmax

# Peak picking (kept for other modes)
threshold_range = [1.4, 1.6]
pre_avg_range   = [25, 30, 35]
post_avg_range  = [25, 30, 35]
pre_max_range   = [1]
post_max_range  = [1]

# Evaluation ranges
global_shift_range = [0.070]
double_onset_correction_range = [0.10]
eval_window_range = [0.05]

# =========================
# DEFAULTS (aligned with your working setup)
# =========================
num_bands = 30
fmin = 1000
fmax = 14000
fref = 2800

threshold = 1.4
pre_avg = 25
post_avg = 25
pre_max = 1
post_max = 1

global_shift = 0.070
double_onset_correction = 0.10

# =========================
# AUDIO FILES
# =========================
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))
print(f"Found {len(list_files)} audio files")

# =========================
# BUILD PARAMETER COMBINATIONS
# =========================
if which_search == 'input_features':  # or 'input_features' or 'peak_picking' or 'evaluation'
    all_combos = product(num_bands_range, fmin_range, fmax_range, fref_range)
    parameter_combinations = [
        (nb, fmn, fmx, frf)
        for (nb, fmn, fmx, frf) in all_combos
        if fmn < frf < fmx
    ]
    print(f"Testing {len(parameter_combinations)} input feature combinations (after constraint filtering)")

elif which_search == 'peak_picking':
    parameter_combinations = list(product(threshold_range, pre_avg_range, post_avg_range, pre_max_range, post_max_range))
    print(f"Testing {len(parameter_combinations)} peak picking combinations")

elif which_search == 'evaluation':
    parameter_combinations = list(product(eval_window_range, global_shift_range, double_onset_correction_range))
    print(f"Testing {len(parameter_combinations)} evaluation combinations")

elif which_search == 'full_search':
    all_combos = product(
        num_bands_range, fmin_range, fmax_range, fref_range,
        threshold_range, pre_avg_range, post_avg_range, pre_max_range, post_max_range,
        global_shift_range, double_onset_correction_range
    )
    parameter_combinations = [
        (nb, fmn, fmx, frf, thr, pav, poav, pmax, pomax, gsh, doc)
        for (nb, fmn, fmx, frf, thr, pav, poav, pmax, pomax, gsh, doc) in all_combos
        if fmn < frf < fmx
    ]
    print(f"Testing {len(parameter_combinations)} full parameter combinations")
    print("WARNING: This might take a very long time!")

# =========================
# GRID SEARCH LOOP
# =========================
overall_fmeasure_and_parameters = {}
rows_for_csv = []  # to export a per-combination CSV
best_fmeasure = 0.0
best_params = None

print(f"Starting grid search at {datetime.now().strftime('%H:%M:%S')}")

for i in tqdm(range(len(parameter_combinations)), desc="Grid Search Progress"):

    # Start from defaults, then override with the current combination
    cur_num_bands = num_bands
    cur_fmin = fmin
    cur_fmax = fmax
    cur_fref = fref

    cur_thr = threshold
    cur_pre_avg = pre_avg
    cur_post_avg = post_avg
    cur_pre_max = pre_max
    cur_post_max = post_max

    cur_global_shift = global_shift
    cur_double_corr = double_onset_correction
    cur_eval_window = EVAL_WINDOW

    if which_search == 'input_features':
        cur_num_bands, cur_fmin, cur_fmax, cur_fref = parameter_combinations[i]

    elif which_search == 'peak_picking':
        cur_thr, cur_pre_avg, cur_post_avg, cur_pre_max, cur_post_max = parameter_combinations[i]

    elif which_search == 'evaluation':
        cur_eval_window, cur_global_shift, cur_double_corr = parameter_combinations[i]

    elif which_search == 'full_search':
        (cur_num_bands, cur_fmin, cur_fmax, cur_fref,
         cur_thr, cur_pre_avg, cur_post_avg, cur_pre_max, cur_post_max,
         cur_global_shift, cur_double_corr) = parameter_combinations[i]

    # Sanity constraint (should already be enforced for feature searches)
    if not (cur_fmin < cur_fref < cur_fmax):
        continue

    # Metrics collected for this combination
    list_fscores_in_set = []
    list_precision_in_set = []
    list_recall_in_set = []
    list_n_events_in_set = []

    errors = 0  # diagnostics

    # Process each audio file
    for wav_path in list_files:
        try:
            # Ground truth file must sit next to the .wav (same name, .txt extension)
            gt_path = wav_path.replace('.wav', '.txt')
            if not os.path.exists(gt_path):
                print(f"[MISS] GT not found: {gt_path}")
                errors += 1
                continue
            gt_onsets = my_eval.get_reference_onsets(gt_path)

            # Experiment window from metadata (match on full filename with extension)
            base_name = os.path.basename(wav_path)
            row = metadata.loc[metadata['Filename'] == base_name]
            if row.empty:
                print(f"[SKIP] '{base_name}': not in metadata (match must include .wav).")
                errors += 1
                continue

            exp_start = row['Start_experiment_sec'].values[0]
            exp_end = row['End_experiment_sec'].values[0]
            if pd.isna(exp_start) or pd.isna(exp_end) or exp_end <= exp_start:
                print(f"[SKIP] '{base_name}': invalid start/end in metadata ({exp_start}–{exp_end}).")
                errors += 1
                continue

            # --- ROBUST UNPACKING ---
            # Some versions of high_frequency_content may return (pred, act)
            # while others may return (pred, act, extra) e.g., debug/figure.
            result = onset_detectors.high_frequency_content(
                wav_path,
                hop_length=hop_length,
                sr=sr,
                spec_num_bands=cur_num_bands,
                spec_fmin=cur_fmin,
                spec_fmax=cur_fmax,
                spec_fref=cur_fref,
                pp_threshold=cur_thr,
                pp_pre_avg=cur_pre_avg,
                pp_post_avg=cur_post_avg,
                pp_pre_max=cur_pre_max,
                pp_post_max=cur_post_max,
                visualise_activation=False  # disable for speed
            )
            if isinstance(result, tuple):
                if len(result) < 2:
                    raise ValueError("high_frequency_content returned <2 outputs.")
                pred_sec, activation_frames = result[0], result[1]
            else:
                # Fallback: assume only predictions were returned
                pred_sec, activation_frames = result, np.array([])

            # Keep only events inside the experiment window
            gt_onsets, pred_sec, activation_frames = my_eval.discard_events_outside_experiment_window(
                exp_start, exp_end, gt_onsets, pred_sec, activation_frames,
                hop_length=hop_length, sr=sr
            )

            # Apply the same corrections as in your eval script
            pred_sec = my_eval.global_shift_correction(pred_sec, cur_global_shift)
            pred_sec = my_eval.double_onset_correction(pred_sec, correction=cur_double_corr)

            # Metrics
            F1, P, R, _, _, _ = f_measure(gt_onsets, pred_sec, window=cur_eval_window)

            list_fscores_in_set.append(F1)
            list_precision_in_set.append(P)
            list_recall_in_set.append(R)
            list_n_events_in_set.append(len(gt_onsets))

        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            errors += 1
            continue

    # Skip this combination if no valid files were processed
    if not list_fscores_in_set:
        print(f"[COMBO {i}] processed=0 errors={errors}")
        # Also append to CSV with NaNs so you can see failures
        rows_for_csv.append({
            'combo_id': i, 'f_measure': np.nan, 'precision': np.nan, 'recall': np.nan,
            'num_bands': cur_num_bands, 'fmin': cur_fmin, 'fmax': cur_fmax, 'fref': cur_fref,
            'threshold': cur_thr, 'pre_avg': cur_pre_avg, 'post_avg': cur_post_avg,
            'pre_max': cur_pre_max, 'post_max': cur_post_max,
            'global_shift': cur_global_shift, 'double_onset_correction': cur_double_corr,
            'eval_window': cur_eval_window, 'n_files_processed': 0, 'errors': errors
        })
        continue

    # Weighted averages (weighted by # of GT events)
    overall_f = my_eval.compute_weighted_average(list_fscores_in_set, list_n_events_in_set)
    overall_p = my_eval.compute_weighted_average(list_precision_in_set, list_n_events_in_set)
    overall_r = my_eval.compute_weighted_average(list_recall_in_set, list_n_events_in_set)

    # Store combination result
    overall_fmeasure_and_parameters[i] = {
        'f_measure': overall_f,
        'precision': overall_p,
        'recall': overall_r,
        'num_bands': cur_num_bands,
        'fmin': cur_fmin,
        'fmax': cur_fmax,
        'fref': cur_fref,
        'threshold': cur_thr,
        'pre_avg': cur_pre_avg,
        'post_avg': cur_post_avg,
        'pre_max': cur_pre_max,
        'post_max': cur_post_max,
        'global_shift': cur_global_shift,
        'double_onset_correction': cur_double_corr,
        'eval_window': cur_eval_window,
        'n_files_processed': len(list_fscores_in_set)
    }

    # Row for CSV
    rows_for_csv.append({
        'combo_id': i, 'f_measure': overall_f, 'precision': overall_p, 'recall': overall_r,
        'num_bands': cur_num_bands, 'fmin': cur_fmin, 'fmax': cur_fmax, 'fref': cur_fref,
        'threshold': cur_thr, 'pre_avg': cur_pre_avg, 'post_avg': cur_post_avg,
        'pre_max': cur_pre_max, 'post_max': cur_post_max,
        'global_shift': cur_global_shift, 'double_onset_correction': cur_double_corr,
        'eval_window': cur_eval_window, 'n_files_processed': len(list_fscores_in_set), 'errors': errors
    })

    # Track best
    if overall_f > best_fmeasure:
        best_fmeasure = overall_f
        best_params = overall_fmeasure_and_parameters[i].copy()
        print(f"\nNew best F-measure: {best_fmeasure:.4f}")
        print(f"Parameters: {best_params}")

    print(f"[COMBO {i}] processed={len(list_fscores_in_set)} errors={errors}")

# =========================
# SAVE RESULTS (JSON + CSV)
# =========================
with open(output_file, "w") as f:
    json.dump(overall_fmeasure_and_parameters, f, indent=2)

best_results_file = output_file.replace('.json', '_best_results.json')
summary_results = {
    'best_combination': best_params if best_params is not None else {},
    'search_type': which_search,
    'total_combinations_tested': len(parameter_combinations),
    'total_files_processed': len(list_files),
    'search_completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open(best_results_file, "w") as f:
    json.dump(summary_results, f, indent=2)

# CSV export of every tested combination (including failures with NaNs)
pd.DataFrame(rows_for_csv).to_csv(csv_summary, index=False)

# =========================
# FINAL LOG
# =========================
print("\n" + "="*60)
print("GRID SEARCH COMPLETED")
print("="*60)
print(f"Search type: {which_search}")
print(f"Total combinations tested: {len(parameter_combinations)}")
print(f"Best F-measure: {best_fmeasure:.4f}")
if best_params is None:
    print("No combination produced valid results. Check paths, metadata and .txt files.")
else:
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
print(f"Results saved to: {output_file}")
print(f"Best results summary saved to: {best_results_file}")
print(f"CSV summary saved to: {csv_summary}")
print(f"Completed at: {datetime.now().strftime('%H:%M:%S')}")
