import os
import glob
import json
from itertools import product
from datetime import datetime
import tempfile
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
from  src.detection import onset_detection_algorithms as onset_detectors
from src.detection import evaluation as my_eval
from mir_eval_modified.onset import f_measure

# =========================
# CONFIG
# =========================
EVAL_WINDOW = 0.05  # default eval window (overridden only in 'evaluation' search)

audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\Normalised_Audio\Data_grid_search'
metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\metadata_testing_1.csv'
output_directory = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\output_grid_search_noise_robust'

os.makedirs(output_directory, exist_ok=True)

# =========================
# NOISE CONFIGURATION
# =========================
# Enable/disable noise testing
test_noise_conditions = True  # Set to False to test only clean audio

# Noise types to test
noise_types = ['white', 'pink', 'brown']  # Remove 'environmental' if no real noise samples

# SNR levels in dB (higher = less noise)
noise_levels_snr_db = [30, 20, 15, 10, 5]

# Environmental noise folder (optional)
environmental_noise_folder = None  # Set path if you have real environmental noise samples
# environmental_noise_folder = r'C:\path\to\environmental\noise\samples'

# Include environmental noise if folder exists
if environmental_noise_folder and os.path.exists(environmental_noise_folder):
    environmental_noise_files = glob.glob(os.path.join(environmental_noise_folder, "*.wav"))
    if environmental_noise_files:
        noise_types.append('environmental')

# Which search to run:
# 'input_features', 'peak_picking', 'evaluation', 'full_search'
which_search = 'evaluation'  # Start with evaluation search for faster testing

# Output files
base_name = f"HFC_{which_search}_noise_robust" if test_noise_conditions else f"HFC_{which_search}_clean_only"
output_file = os.path.join(output_directory, f"{base_name}.json")
csv_summary = os.path.join(output_directory, f"{base_name}_summary.csv")
csv_detailed = os.path.join(output_directory, f"{base_name}_detailed_results.csv")

# Fixed HFC frontend parameters
hop_length = 441
sr = 44100

# =========================
# NOISE GENERATION FUNCTIONS
# =========================
def generate_colored_noise(duration, sr, noise_type='white'):
    """Generate different types of colored noise"""
    samples = int(duration * sr)
    
    if noise_type == 'white':
        return np.random.normal(0, 1, samples)
    
    elif noise_type == 'pink':
        # Pink noise (1/f noise)
        white_noise = np.random.normal(0, 1, samples)
        fft_white = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(len(fft_white), 1/sr)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[0] = pink_filter[1]  # Fix DC component
        fft_pink = fft_white * pink_filter
        return np.real(np.fft.ifft(fft_pink))
    
    elif noise_type == 'brown':
        # Brown noise (1/f^2 noise)
        white_noise = np.random.normal(0, 1, samples)
        fft_white = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(len(fft_white), 1/sr)
        freqs[0] = 1
        brown_filter = 1 / np.abs(freqs)
        brown_filter[0] = brown_filter[1]
        fft_brown = fft_white * brown_filter
        return np.real(np.fft.ifft(fft_brown))
    
    else:
        return np.random.normal(0, 1, samples)

def load_environmental_noise(duration, sr):
    """Load and prepare environmental noise samples"""
    if not environmental_noise_folder or not environmental_noise_files:
        return generate_colored_noise(duration, sr, 'white')
    
    # Randomly select a noise file
    selected_noise_file = np.random.choice(environmental_noise_files)
    noise_audio, _ = librosa.load(selected_noise_file, sr=sr)
    
    # Repeat or trim to match desired duration
    target_samples = int(duration * sr)
    if len(noise_audio) < target_samples:
        repetitions = int(np.ceil(target_samples / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repetitions)
    
    return noise_audio[:target_samples]

def add_noise_to_audio(audio, noise_type, snr_db, sr):
    """Add noise to audio signal with specified SNR"""
    duration = len(audio) / sr
    
    if noise_type == 'environmental':
        noise = load_environmental_noise(duration, sr)
    else:
        noise = generate_colored_noise(duration, sr, noise_type)
    
    # Ensure same length
    min_length = min(len(audio), len(noise))
    audio = audio[:min_length]
    noise = noise[:min_length]
    
    # Calculate signal and noise power
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate noise scaling factor for desired SNR
    if noise_power == 0:
        return audio  # No noise to add
    
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
    
    # Add scaled noise to signal
    noisy_audio = audio + noise_scale * noise
    
    return noisy_audio

def create_noise_conditions():
    """Create list of all noise conditions to test"""
    conditions = [{'name': 'clean', 'noise_type': None, 'snr_db': None}]
    
    if test_noise_conditions:
        for noise_type in noise_types:
            for snr_db in noise_levels_snr_db:
                conditions.append({
                    'name': f"{noise_type}_snr{snr_db}db",
                    'noise_type': noise_type,
                    'snr_db': snr_db
                })
    
    return conditions

# =========================
# LOAD METADATA
# =========================
metadata = pd.read_csv(metadata_path)
for col in ['Filename', 'Start_experiment_sec', 'End_experiment_sec']:
    if col not in metadata.columns:
        raise ValueError(f"Missing column '{col}' in metadata CSV.")
metadata['Start_experiment_sec'] = pd.to_numeric(metadata['Start_experiment_sec'], errors='coerce')
metadata['End_experiment_sec'] = pd.to_numeric(metadata['End_experiment_sec'], errors='coerce')

# =========================
# PARAMETER RANGES
# =========================
# Input features
num_bands_range = [30]
fmin_range = [1000]
fmax_range = [14000]
fref_range = [2800]

# Peak picking
threshold_range = [1.4]
pre_avg_range = [25]
post_avg_range = [25]
pre_max_range = [1]
post_max_range = [1]

# Evaluation ranges
global_shift_range = [0.070]
double_onset_correction_range = [0.10]
eval_window_range = [0.05]

# Defaults
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
# SETUP
# =========================
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))
noise_conditions = create_noise_conditions()

print(f"Found {len(list_files)} audio files")
print(f"Testing {len(noise_conditions)} noise conditions: {[c['name'] for c in noise_conditions]}")

# Build parameter combinations
if which_search == 'input_features':
    all_combos = product(num_bands_range, fmin_range, fmax_range, fref_range)
    parameter_combinations = [(nb, fmn, fmx, frf) for (nb, fmn, fmx, frf) in all_combos if fmn < frf < fmx]
elif which_search == 'peak_picking':
    parameter_combinations = list(product(threshold_range, pre_avg_range, post_avg_range, pre_max_range, post_max_range))
elif which_search == 'evaluation':
    parameter_combinations = list(product(eval_window_range, global_shift_range, double_onset_correction_range))
elif which_search == 'full_search':
    all_combos = product(
        num_bands_range, fmin_range, fmax_range, fref_range,
        threshold_range, pre_avg_range, post_avg_range, pre_max_range, post_max_range,
        global_shift_range, double_onset_correction_range
    )
    parameter_combinations = [(nb, fmn, fmx, frf, thr, pav, poav, pmax, pomax, gsh, doc) 
                             for (nb, fmn, fmx, frf, thr, pav, poav, pmax, pomax, gsh, doc) in all_combos 
                             if fmn < frf < fmx]

print(f"Testing {len(parameter_combinations)} parameter combinations")
if test_noise_conditions:
    print(f"Total tests: {len(parameter_combinations)} × {len(noise_conditions)} = {len(parameter_combinations) * len(noise_conditions)}")

# =========================
# GRID SEARCH WITH NOISE ANALYSIS
# =========================
overall_results = {}
detailed_results = []  # For CSV export
best_results_by_condition = {}

print(f"Starting noise-robust grid search at {datetime.now().strftime('%H:%M:%S')}")

# Set random seed for reproducible noise
np.random.seed(42)

for combo_idx in tqdm(range(len(parameter_combinations)), desc="Parameter Combinations"):
    
    # Parse current combination
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
        cur_num_bands, cur_fmin, cur_fmax, cur_fref = parameter_combinations[combo_idx]
    elif which_search == 'peak_picking':
        cur_thr, cur_pre_avg, cur_post_avg, cur_pre_max, cur_post_max = parameter_combinations[combo_idx]
    elif which_search == 'evaluation':
        cur_eval_window, cur_global_shift, cur_double_corr = parameter_combinations[combo_idx]
    elif which_search == 'full_search':
        (cur_num_bands, cur_fmin, cur_fmax, cur_fref, cur_thr, cur_pre_avg, cur_post_avg, 
         cur_pre_max, cur_post_max, cur_global_shift, cur_double_corr) = parameter_combinations[combo_idx]

    if not (cur_fmin < cur_fref < cur_fmax):
        continue

    # Results for this parameter combination across all noise conditions
    combo_results = {}
    
    # Test each noise condition
    for condition in tqdm(noise_conditions, desc="Noise Conditions", leave=False):
        condition_name = condition['name']
        
        # Metrics for this condition
        condition_fscores = []
        condition_precisions = []
        condition_recalls = []
        condition_n_events = []
        
        temp_files_to_cleanup = []  # Track temporary files
        
        # Process each audio file
        for wav_path in list_files:
            try:
                # Check ground truth and metadata
                gt_path = wav_path.replace('.wav', '.txt')
                if not os.path.exists(gt_path):
                    continue
                
                gt_onsets = my_eval.get_reference_onsets(gt_path)
                base_name = os.path.basename(wav_path)
                row = metadata.loc[metadata['Filename'] == base_name]
                if row.empty:
                    continue
                
                exp_start = row['Start_experiment_sec'].values[0]
                exp_end = row['End_experiment_sec'].values[0]
                if pd.isna(exp_start) or pd.isna(exp_end) or exp_end <= exp_start:
                    continue

                # Prepare audio file for processing
                audio_file_to_process = wav_path  # Default to original
                
                if condition_name != 'clean':
                    # Load original audio and add noise
                    audio, sr_loaded = librosa.load(wav_path, sr=sr)
                    noisy_audio = add_noise_to_audio(audio, condition['noise_type'], condition['snr_db'], sr)
                    
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    sf.write(temp_file.name, noisy_audio, sr)
                    audio_file_to_process = temp_file.name
                    temp_files_to_cleanup.append(temp_file.name)

                # Run HFC onset detection
                result = onset_detectors.high_frequency_content(
                    audio_file_to_process,
                    hop_length=hop_length, sr=sr,
                    spec_num_bands=cur_num_bands, spec_fmin=cur_fmin, spec_fmax=cur_fmax, spec_fref=cur_fref,
                    pp_threshold=cur_thr, pp_pre_avg=cur_pre_avg, pp_post_avg=cur_post_avg,
                    pp_pre_max=cur_pre_max, pp_post_max=cur_post_max,
                    visualise_activation=False
                )
                
                if isinstance(result, tuple) and len(result) >= 2:
                    pred_sec, activation_frames = result[0], result[1]
                else:
                    pred_sec, activation_frames = result, np.array([])

                # Filter to experiment window
                gt_onsets, pred_sec, activation_frames = my_eval.discard_events_outside_experiment_window(
                    exp_start, exp_end, gt_onsets, pred_sec, activation_frames,
                    hop_length=hop_length, sr=sr
                )

                # Apply corrections
                pred_sec = my_eval.global_shift_correction(pred_sec, cur_global_shift)
                pred_sec = my_eval.double_onset_correction(pred_sec, correction=cur_double_corr)

                # Calculate metrics
                F1, P, R, _, _, _ = f_measure(gt_onsets, pred_sec, window=cur_eval_window)

                condition_fscores.append(F1)
                condition_precisions.append(P)
                condition_recalls.append(R)
                condition_n_events.append(len(gt_onsets))

            except Exception as e:
                print(f"Error processing {wav_path} with {condition_name}: {str(e)}")
                continue
        
        # Clean up temporary files
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except:
                pass

        # Calculate aggregate metrics for this condition
        if condition_fscores:
            avg_f = my_eval.compute_weighted_average(condition_fscores, condition_n_events)
            avg_p = my_eval.compute_weighted_average(condition_precisions, condition_n_events)
            avg_r = my_eval.compute_weighted_average(condition_recalls, condition_n_events)
            
            combo_results[condition_name] = {
                'f_measure': avg_f,
                'precision': avg_p,
                'recall': avg_r,
                'n_files': len(condition_fscores),
                'total_events': sum(condition_n_events)
            }
            
            # Track best results for each condition
            if (condition_name not in best_results_by_condition or 
                avg_f > best_results_by_condition[condition_name]['f_measure']):
                best_results_by_condition[condition_name] = {
                    'f_measure': avg_f,
                    'precision': avg_p,
                    'recall': avg_r,
                    'parameters': {
                        'num_bands': cur_num_bands, 'fmin': cur_fmin, 'fmax': cur_fmax, 'fref': cur_fref,
                        'threshold': cur_thr, 'pre_avg': cur_pre_avg, 'post_avg': cur_post_avg,
                        'pre_max': cur_pre_max, 'post_max': cur_post_max,
                        'global_shift': cur_global_shift, 'double_onset_correction': cur_double_corr,
                        'eval_window': cur_eval_window
                    },
                    'combo_id': combo_idx
                }
            
            # Add to detailed results for CSV
            detailed_results.append({
                'combo_id': combo_idx,
                'condition': condition_name,
                'noise_type': condition.get('noise_type', 'none'),
                'snr_db': condition.get('snr_db', 'N/A'),
                'f_measure': avg_f,
                'precision': avg_p,
                'recall': avg_r,
                'n_files': len(condition_fscores),
                'total_events': sum(condition_n_events),
                'num_bands': cur_num_bands, 'fmin': cur_fmin, 'fmax': cur_fmax, 'fref': cur_fref,
                'threshold': cur_thr, 'pre_avg': cur_pre_avg, 'post_avg': cur_post_avg,
                'pre_max': cur_pre_max, 'post_max': cur_post_max,
                'global_shift': cur_global_shift, 'double_onset_correction': cur_double_corr,
                'eval_window': cur_eval_window
            })

    # Store combination results
    overall_results[combo_idx] = {
        'parameters': {
            'num_bands': cur_num_bands, 'fmin': cur_fmin, 'fmax': cur_fmax, 'fref': cur_fref,
            'threshold': cur_thr, 'pre_avg': cur_pre_avg, 'post_avg': cur_post_avg,
            'pre_max': cur_pre_max, 'post_max': cur_post_max,
            'global_shift': cur_global_shift, 'double_onset_correction': cur_double_corr,
            'eval_window': cur_eval_window
        },
        'results_by_condition': combo_results
    }

# =========================
# SAVE RESULTS
# =========================
# Main results file
with open(output_file, "w") as f:
    json.dump(overall_results, f, indent=2)

# Best results by condition
best_results_file = output_file.replace('.json', '_best_by_condition.json')
with open(best_results_file, "w") as f:
    json.dump({
        'best_results_by_condition': best_results_by_condition,
        'search_type': which_search,
        'noise_conditions_tested': [c['name'] for c in noise_conditions],
        'total_combinations_tested': len(parameter_combinations),
        'total_files_processed': len(list_files),
        'search_completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2)

# Detailed results CSV
pd.DataFrame(detailed_results).to_csv(csv_detailed, index=False)

# Summary CSV (best result for each condition)
summary_rows = []
for condition_name, best_result in best_results_by_condition.items():
    row = {
        'condition': condition_name,
        'best_f_measure': best_result['f_measure'],
        'best_precision': best_result['precision'],
        'best_recall': best_result['recall']
    }
    row.update(best_result['parameters'])
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(csv_summary, index=False)

# =========================
# FINAL REPORT
# =========================
print("\n" + "="*80)
print("NOISE-ROBUST GRID SEARCH COMPLETED")
print("="*80)
print(f"Search type: {which_search}")
print(f"Total parameter combinations: {len(parameter_combinations)}")
print(f"Noise conditions tested: {len(noise_conditions)}")

print(f"\nBest F-measure by condition:")
for condition_name in sorted(best_results_by_condition.keys()):
    result = best_results_by_condition[condition_name]
    print(f"  {condition_name:20s}: {result['f_measure']:.4f}")

print(f"\nFiles saved:")
print(f"  Main results: {output_file}")
print(f"  Best by condition: {best_results_file}")
print(f"  Detailed CSV: {csv_detailed}")
print(f"  Summary CSV: {csv_summary}")
print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")

# Performance degradation analysis
if test_noise_conditions and 'clean' in best_results_by_condition:
    print(f"\nPerformance degradation analysis:")
    clean_f1 = best_results_by_condition['clean']['f_measure']
    print(f"  Clean audio F1: {clean_f1:.4f}")
    
    for condition_name, result in best_results_by_condition.items():
        if condition_name != 'clean' and 'snr' in condition_name:
            degradation = ((clean_f1 - result['f_measure']) / clean_f1) * 100
            print(f"  {condition_name:20s}: {result['f_measure']:.4f} (-{degradation:.1f}%)")