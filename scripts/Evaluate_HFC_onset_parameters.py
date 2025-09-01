import os
import glob
from  src.detection import onset_detection_algorithms as onset_detectors
from src.detection import onsets_offsets_detection_utils as onset_utils
from src.detection import evaluation as my_eval
from src.detection.save_results import save_results_in_csv, save_global_results_latex
from src.analysis.visualization import plot_signal, visualize_activation_and_gt
from tqdm import tqdm
import pandas as pd
from mir_eval_modified.onset import f_measure
import json


import matplotlib.pyplot as plt

# WHEN running with parameters different from default, change here and name the results folder with the new parameters
EVAL_WINDOW = 0.1

# Parameters for the HFC onset detection function
HFC_parameters = {
    'hop_length': 441, 
    'sr': 44100, 
    'spec_num_bands': 15, 
    'spec_fmin': 2500, 
    'spec_fmax': 5000, 
    'spec_fref': 2800,
    'pp_threshold': 1.8, 
    'pp_pre_avg': 25, 
    'pp_post_avg': 1, 
    'pp_pre_max': 3, 
    'pp_post_max': 2,
    'global shift': 0.1, 
    'double_onset_correction': 0.1
}

# File paths configuration
audio_folder = 'C:\\Users\\anton\\Test_VPA_normalised\\Data'
metadata = pd.read_csv("C:\\Users\\anton\\Test_VPA_normalised\\metadata_vpa_testing.csv")
save_evaluation_results_path = r'C:\Users\anton\Test_VPA_normalised\counting_calls_results_on_testing_HFC_only'

# Create results directory if it doesn't exist
if not os.path.exists(save_evaluation_results_path):
    os.makedirs(save_evaluation_results_path)

# Get list of audio files
list_files = glob.glob(os.path.join(audio_folder, "*.wav"))

# Initialize list to collect results
results = []

print(f"Processing {len(list_files)} files with HFC onset detection...")

for file in tqdm(list_files, desc="Processing files"):
    # Get filename without extension
    filename = os.path.basename(file)[:-4]
    
    # Create folder for each chick results
    chick_folder = os.path.join(save_evaluation_results_path, filename)
    if not os.path.exists(chick_folder):
        os.makedirs(chick_folder)

    # Get ground truth onsets
    gt_onsets = my_eval.get_reference_onsets(file.replace('.wav', '.txt'))
    
    # Get experiment start and end times from metadata
    exp_start = float(metadata[metadata['Filename'] == filename]['Start_experiment_sec'].values[0])
    exp_end = float(metadata[metadata['Filename'] == filename]['End_experiment_sec'].values[0])
    
    # Get metadata for this file
    file_group = metadata[metadata['Filename'] == filename]['Group'].values[0]
    file_sex = metadata[metadata['Filename'] == filename]['Sex'].values[0]

    # High Frequency Content onset detection
    HFC_results_folder = os.path.join(chick_folder, f'HFC_default_parameters_window{EVAL_WINDOW}')
    if not os.path.exists(HFC_results_folder):
        os.makedirs(HFC_results_folder)
    

        # Extract HFC parameters (exclude correction parameters for initial detection)
    hfc_params = {k: v for k, v in HFC_parameters.items() if k not in ['global_shift', 'double_onset_correction']}
    # Run HFC onset detection
    hfc_pred_scnd, HFC_activation_frames = onset_detectors.high_frequency_content(file, visualise_activation=True, **hfc_params)

    # Filter events within experiment window
    gt_onsets, hfc_pred_scnd, HFC_activation_frames = my_eval.discard_events_outside_experiment_window(
        exp_start, exp_end, gt_onsets, hfc_pred_scnd, HFC_activation_frames, 
        hop_length=441, sr=44100
    )
    
    # Apply corrections
    hfc_pred_scnd = my_eval.global_shift_correction(hfc_pred_scnd, HFC_parameters['global shift'])
    hfc_pred_scnd = my_eval.double_onset_correction(hfc_pred_scnd, correction=HFC_parameters['double_onset_correction'])

    # Visualize results
    visualize_activation_and_gt(
        plot_dir=HFC_results_folder,
        file_name=os.path.basename(file),
        onset_detection_funtion_name='High frequency content',
        gt_onsets=gt_onsets, 
        activation=HFC_activation_frames, 
        start_exp=exp_start, 
        end_exp=exp_end, 
        hop_length=441, 
        sr=44100
    )
    
    # Save predictions to file
    hfc_predictions_seconds_df = pd.DataFrame(hfc_pred_scnd, columns=['onset_seconds'])
    hfc_predictions_seconds_df.to_csv(
        os.path.join(HFC_results_folder, f'{filename}_HFCpredictions.csv'), 
        index=False
    )

    # Count detected calls
    calls_detected_with_hfc = len(hfc_pred_scnd)
    
    # Add results to collection
    results.append({
        'Filename': filename,
        'Calls_detected': calls_detected_with_hfc,
        'Algorithm': 'HFC',
        'Group': file_group,
        'Sex': file_sex
    })

# Create DataFrame from results
calls_detected_df = pd.DataFrame(results)

# Save results
print("Saving results...")

# Save global calls detected in CSV
calls_detected_df.to_csv(os.path.join(save_evaluation_results_path, 'Calls_detected_HFC.csv'), index=False)

# Compute and save average calls detected per group
if 'Group' in calls_detected_df.columns:
    average_calls_per_group = calls_detected_df.groupby('Group')['Calls_detected'].mean()
    average_calls_per_group.to_csv(os.path.join(save_evaluation_results_path, 'Average_calls_detected_per_group.csv'))
    
    # Plot average calls per group
    plt.figure(figsize=(10, 6))
    average_calls_per_group.plot(kind='bar', title='Average HFC calls detected per group')
    plt.ylabel('Average calls detected')
    plt.xlabel('Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_evaluation_results_path, 'Average_calls_detected_per_group.png'))
    plt.close()

# Compute and save average calls detected per sex
if 'Sex' in calls_detected_df.columns:
    average_calls_per_sex = calls_detected_df.groupby('Sex')['Calls_detected'].mean()
    average_calls_per_sex.to_csv(os.path.join(save_evaluation_results_path, 'Average_calls_detected_per_sex.csv'))
    
    # Plot average calls per sex
    plt.figure(figsize=(8, 6))
    average_calls_per_sex.plot(kind='bar', title='Average HFC calls detected per sex')
    plt.ylabel('Average calls detected')
    plt.xlabel('Sex')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_evaluation_results_path, 'Average_calls_detected_per_sex.png'))
    plt.close()

# Compute and save overall average calls detected
average_calls_detected = calls_detected_df['Calls_detected'].mean()
overall_stats = pd.DataFrame({
    'Algorithm': ['HFC'],
    'Average_calls_detected': [average_calls_detected],
    'Total_files_processed': [len(calls_detected_df)],
    'Min_calls_detected': [calls_detected_df['Calls_detected'].min()],
    'Max_calls_detected': [calls_detected_df['Calls_detected'].max()],
    'Std_calls_detected': [calls_detected_df['Calls_detected'].std()]
})
overall_stats.to_csv(os.path.join(save_evaluation_results_path, 'Overall_HFC_statistics.csv'), index=False)

# Generate detailed evaluation results if functions are available
try:
    individual_performances_chicks = save_results_in_csv(save_evaluation_results_path)
    print("Individual performance results saved successfully.")
except Exception as e:
    print(f"Warning: Could not save individual performance results: {e}")

try:
    table_csv, latex_table = save_global_results_latex(save_evaluation_results_path)
    print("Global results and LaTeX table saved successfully.")
except Exception as e:
    print(f"Warning: Could not save global results: {e}")

print(f"Processing complete! Results saved to: {save_evaluation_results_path}")
print(f"Total files processed: {len(list_files)}")
print(f"Average calls detected per file: {average_calls_detected:.2f}")
print('Done!')