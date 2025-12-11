import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import onsets_offsets_detection_utils as onset_detectors
from onsets_offsets_detection_utils import high_frequency_content, offset_detection_first_order
from utils_interaction_analysis import load_stimulus_log_absolute, filter_calls_produced
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Paths
# audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2'
# stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2\Stimuli_log'
# metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2\metadata_testing.csv'

# audio_folder = r'C:\Users\anton\OneDrive\Documenti\Automatic_testing\Normalised_audio'
# stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Automatic_testing\Stimuli_log'
# metadata_path = r'C:\Users\anton\OneDrive\Documenti\Automatic_testing\metadata_testing_automatic.csv'


audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_automatic\Testing_automatic'
stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_automatic\Stimuli_log'
metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_automatic\metadata_automatic_testing.csv'

# Output
save_results_path = os.path.join(audio_folder, 'analysis_results_latency')
os.makedirs(save_results_path, exist_ok=True)

# Constants
EXPERIMENT_DURATION = 600  # 10 minutes in seconds - adjust based on your files

# Load metadata
metadata = pd.read_csv(metadata_path)
metadata.columns = metadata.columns.str.strip()

# Normalize column names
for col in ['Chick_ID', 'Filename', 'Condition', 'Sex']:
    if col in metadata.columns:
        metadata[col] = metadata[col].astype(str).str.strip()

# HFC parameters for call detection
# HFC_parameters = {
#     'hop_length': 441, 'sr': 44100, 'spec_num_bands': 30, 'spec_fmin': 1000, 'spec_fmax': 14000, 'spec_fref': 2800,
#     'pp_threshold': 1.4, 'pp_pre_avg': 25, 'pp_post_avg': 1, 'pp_pre_max': 1, 'pp_post_max': 1,
#     'global_shift': 0.070, 'double_onset_correction': 0.1
# }



HFC_parameters = {
    'hop_length': 441, 'sr': 44100, 'spec_num_bands': 24, 'spec_fmin': 1500, 'spec_fmax': 10000, 'spec_fref': 3000,
    'pp_threshold': 0.5, 'pp_pre_avg': 25, 'pp_post_avg': 1, 'pp_pre_max': 1, 'pp_post_max': 1,
    'global_shift': 0.070, 'double_onset_correction': 0.1
}

# ADDED: Stimulus durations for feedback analysis
STIMULUS_DURATIONS = {
    'pleasure': {'pleasure': 0.107, 'white': 0.107, 'silence': 0.107},
    'contact': {'contact': 0.223, 'white': 0.223, 'silence': 0.223},
    'cluck': {'cluck': 0.107, 'white': 0.107, 'silence': 0.107}
}

# Main processing loop
all_calls_data = []
# ADDED: Lists for new analyses
calls_during_feedback = []
response_latencies = []

audio_files = glob.glob(os.path.join(audio_folder, '*.wav'))

for audio_file in tqdm(audio_files, desc="Processing audio files"):
    base_name = os.path.basename(audio_file)
    
    # Find matching metadata by exact filename match
    meta_row = metadata[metadata['Filename'] == base_name]
    
    if meta_row.empty:
        print(f"[SKIP] No metadata found for file: {base_name}")
        continue

    meta_row = meta_row.iloc[0]
    
    # Extract ChickID cleanly from Filename (remove .wav extension)
    chick_id = os.path.splitext(meta_row['Filename'])[0]
    
    recorded_start_time = float(meta_row['Start_experiment_sec'])
    biological_condition = meta_row['Condition']
    recorded_end_time = float(meta_row['End_experiment_sec']) if 'End_experiment_sec' in meta_row and pd.notna(meta_row['End_experiment_sec']) else recorded_start_time + EXPERIMENT_DURATION
    sex = meta_row['Sex'] if 'Sex' in meta_row and pd.notna(meta_row['Sex']) else 'Unknown'
    
    # Compute true experimental duration for this file
    experiment_duration = recorded_end_time - recorded_start_time
    # Number of 1-minute bins covering the whole experimental window
    num_bins = int(np.ceil(experiment_duration / 60.0))
    
    # Load stimulus log to filter out calls during stimuli
    stim_log_filename = meta_row['Stimulus_log_filename']
    stim_log_path = os.path.join(stimuli_folder, stim_log_filename)
    
    if not os.path.exists(stim_log_path):
        print(f"[WARNING] Stimulus log not found: {stim_log_path}")
        stimuli = []
    else:
        stimuli = load_stimulus_log_absolute(stim_log_path, recorded_start_time)
        # Keep only stimuli that fall within the experimental window
        stimuli = [
            s for s in stimuli
            if recorded_start_time <= s['time'] <= recorded_end_time
        ]
    
    # Detect calls using HFC method
    hfc_params = {k: v for k, v in HFC_parameters.items() if k not in ['global_shift', 'double_onset_correction']}
    hfc_onsets = onset_detectors.high_frequency_content(audio_file, visualise_activation=False, **hfc_params)
    hfc_onsets = onset_detectors.global_shift_correction(hfc_onsets, HFC_parameters['global_shift'])
    hfc_onsets = onset_detectors.double_onset_correction(hfc_onsets, correction=HFC_parameters['double_onset_correction'])
    
    # Detect offsets
    predicted_offsets = offset_detection_first_order(audio_file, hfc_onsets)

    # Restrict HFC detections to the experimental window for latency/feedback analyses
    window_mask = (hfc_onsets >= recorded_start_time) & (predicted_offsets <= recorded_end_time)
    hfc_onsets_window = hfc_onsets[window_mask]
    predicted_offsets_window = predicted_offsets[window_mask]

    # ADDED: Analysis of calls during feedback and response latency
    if stimuli:
        bio_condition_lower = biological_condition.lower().strip()
        
        for stim_idx, stimulus in enumerate(stimuli):
            stim_type_lower = stimulus['type'].lower().strip()
            stim_time = stimulus['time']
            
            if stim_type_lower == 'start':
                continue
            
            # Get stimulus duration
            if bio_condition_lower in STIMULUS_DURATIONS:
                stim_duration = STIMULUS_DURATIONS[bio_condition_lower].get(stim_type_lower, 0.107)
            else:
                stim_duration = 0.107
            
            stim_end_time = stim_time + stim_duration
            
            # 1. Find calls during feedback
            for call_idx, (onset, offset) in enumerate(zip(hfc_onsets_window, predicted_offsets_window)):
                if onset <= stim_end_time and offset >= stim_time:
                    calls_during_feedback.append({
                        'ChickID': chick_id,
                        'Sex': sex,
                        'Biological_Condition': biological_condition,
                        'Stimulus_Type': stimulus['type'],
                        'Stimulus_Time': stim_time,
                        'Stimulus_End_Time': stim_end_time,
                        'Call_Onset': onset,
                        'Call_Offset': offset,
                        'Call_Duration': offset - onset
                    })
            
            # 2. Find first response after feedback ends
            first_response = None
            for onset in hfc_onsets_window:
                if onset > stim_end_time:
                    first_response = onset
                    break
            
            response_latencies.append({
                'ChickID': chick_id,
                'Sex': sex,
                'Biological_Condition': biological_condition,
                'Stimulus_Type': stimulus['type'],
                'Stimulus_End_Time': stim_end_time,
                'First_Response_Onset': first_response,
                'Response_Latency': first_response - stim_end_time if first_response else None
            })

    # Filter calls produced by chick (exclude calls during stimuli)
    if stimuli:
        calls_produced_onsets, calls_produced_offsets = filter_calls_produced(
            hfc_onsets, predicted_offsets, stimuli, biological_condition
        )
    else:
        calls_produced_onsets = hfc_onsets
        calls_produced_offsets = predicted_offsets

    # Calculate relative times and filter calls within experiment window
    valid_call_count = 0
    for i, (onset, offset) in enumerate(zip(calls_produced_onsets, calls_produced_offsets)):
        # Calculate relative time from experiment start
        onset_rel = onset - recorded_start_time
        offset_rel = offset - recorded_start_time
        
        # Skip calls that start before experiment or end after experiment duration
        if onset_rel < 0 or offset_rel > experiment_duration:
            continue
            
        # Calculate time bin (1-minute bins: bin 1 = 0-60s, bin 2 = 60-120s, etc.)
        time_bin = int(np.floor(onset_rel / 60)) + 1
        
        # Skip calls beyond total number of 1-min bins covering the experiment
        if time_bin > num_bins:
            continue

        valid_call_count += 1
        
        # Create call ID
        call_id = f"{chick_id}_call_{valid_call_count:03d}"

        # Store call data
        call_record = {
            'ChickID': chick_id,
            'Call_ID': call_id,
            'Sex': sex,
            'Biological_Condition': biological_condition,
            'Start_experiment_sec': recorded_start_time,
            'End_experiment_sec': recorded_end_time,
            'Call_Number': valid_call_count,
            'Onset_Absolute_Sec': onset,
            'Offset_Absolute_Sec': offset,
            'Onset_Relative_Sec': onset_rel,
            'Offset_Relative_Sec': offset_rel,
            'Duration_call': offset - onset,
            'Time_Bins': time_bin
        }
        all_calls_data.append(call_record)
    
    print(f"Processed {chick_id}: {len(calls_produced_onsets)} total calls, {valid_call_count} calls within experiment window")

# Create final dataframe
if all_calls_data:
    all_calls_df = pd.DataFrame(all_calls_data)
    
    # Check for calls beyond experiment duration (per-file, using stored start/end)
    over_limit = all_calls_df[
        all_calls_df['Offset_Relative_Sec'] >
        (all_calls_df['End_experiment_sec'] - all_calls_df['Start_experiment_sec'])
    ]
    if not over_limit.empty:
        print("Calls that end after experiment duration:")
        print(over_limit[['ChickID', 'Call_ID', 'Onset_Relative_Sec', 'Offset_Relative_Sec', 'Duration_call']].head())
        print(f"Total calls beyond limit: {len(over_limit)}")
    else:
        print("All calls end within the experiment window.")
    
    # Export main CSV
    all_calls_df.to_csv(os.path.join(save_results_path, "all_chicks_calls.csv"), index=False)
    
    # ADDED: Export new analysis results
    if calls_during_feedback:
        pd.DataFrame(calls_during_feedback).to_csv(os.path.join(save_results_path, "calls_during_feedback.csv"), index=False)
        print(f"Calls during feedback saved: {len(calls_during_feedback)} calls")
    
    if response_latencies:
        pd.DataFrame(response_latencies).to_csv(os.path.join(save_results_path, "response_latencies.csv"), index=False)
        print(f"Response latencies saved: {len(response_latencies)} events")
    
    print(f"\nAnalysis completed!")
    print(f"All calls saved to: all_chicks_calls.csv")
    print(f"Total calls analyzed: {len(all_calls_df)}")
    print(f"Total chicks: {all_calls_df['ChickID'].nunique()}")
    print(f"Time bins covered: {all_calls_df['Time_Bins'].min()} to {all_calls_df['Time_Bins'].max()}")
