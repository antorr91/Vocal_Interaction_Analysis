import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import onsets_offsets_detection_utils as onset_detectors
from onsets_offsets_detection_utils import offset_detection_first_order
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
save_results_path = os.path.join(audio_folder, 'analysis_results_new')
os.makedirs(save_results_path, exist_ok=True)

# Constants
# EXPERIMENT_DURATION = 360  # 6 minutes in seconds - adjust based on your files
# EXPERIMENT_DURATION = 600  # 10 minutes in seconds
# NOTE: Duration is now read directly from metadata (End_experiment_sec - Start_experiment_sec)

# Load metadata
metadata = pd.read_csv(metadata_path)
metadata.columns = metadata.columns.str.strip()

# Normalize column names
for col in ['Chick_ID', 'Filename', 'Condition', 'Sex']:
    if col in metadata.columns:
        metadata[col] = metadata[col].astype(str).str.strip()

# HFC parameters for call detection
# HFC_parameters = {
#     'hop_length': 441, 'sr': 44100, 'spec_num_bands': 32, 'spec_fmin': 1000, 'spec_fmax': 15000, 'spec_fref': 3200,
#     'pp_threshold': 1.4, 'pp_pre_avg': 25, 'pp_post_avg': 40, 'pp_pre_max': 1, 'pp_post_max': 1,
#     'global_shift': 0.10, 'double_onset_correction': 0.1
# }

# SR = 44100
# HOP = 441
# SPEC_NUM_BANDS = 24
# SPEC_FMIN = 1500
# SPEC_FMAX = 10000
# SPEC_FREF = 3000
# PP_THRESHOLD = 0.5
# PP_PRE_AVG = 25
# PP_POST_AVG = 1
# PP_PRE_MAX = 1
# PP_POST_MAX = 1
# GLOBAL_SHIFT = 0.070
# REFRACTORY = 0.1

HFC_parameters = {
    'hop_length': 441, 'sr': 44100, 'spec_num_bands': 24, 'spec_fmin': 1500, 'spec_fmax': 10000, 'spec_fref': 3000,
    'pp_threshold': 0.5, 'pp_pre_avg': 25, 'pp_post_avg': 1, 'pp_pre_max': 1, 'pp_post_max': 1,
    'global_shift': 0.070, 'double_onset_correction': 0.1
}


# Main processing loop
all_calls_data = []
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
    
    # Read experiment timing from metadata
    recorded_start_time = float(meta_row['Start_experiment_sec'])
    recorded_end_time = float(meta_row['End_experiment_sec'])
    biological_condition = meta_row['Condition']
    sex = meta_row['Sex'] if 'Sex' in meta_row and pd.notna(meta_row['Sex']) else 'Unknown'

    # Calculate actual experiment duration from metadata (replaces hardcoded EXPERIMENT_DURATION)
    experiment_duration = recorded_end_time - recorded_start_time
    
    # Load stimulus log to filter out calls during stimuli
    stim_log_filename = meta_row['Stimulus_log_filename']
    stim_log_path = os.path.join(stimuli_folder, stim_log_filename)
    
    if not os.path.exists(stim_log_path):
        print(f"[WARNING] Stimulus log not found: {stim_log_path}")
        stimuli = []
    else:
        stimuli = load_stimulus_log_absolute(stim_log_path, recorded_start_time)
    
    # Detect calls using HFC method
    hfc_params = {k: v for k, v in HFC_parameters.items() if k not in ['global_shift', 'double_onset_correction']}
    hfc_onsets = onset_detectors.high_frequency_content(audio_file, visualise_activation=False, **hfc_params)
    hfc_onsets = onset_detectors.global_shift_correction(hfc_onsets, HFC_parameters['global_shift'])
    hfc_onsets = onset_detectors.double_onset_correction(hfc_onsets, correction=HFC_parameters['double_onset_correction'])
    
    # Detect offsets
    predicted_offsets = offset_detection_first_order(audio_file, hfc_onsets)

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
        # NOW USING experiment_duration FROM METADATA instead of hardcoded EXPERIMENT_DURATION
        if onset_rel < 0 or offset_rel > experiment_duration:
            continue
            
        # Calculate time bin (1-minute bins: bin 1 = 0-60s, bin 2 = 60-120s, etc.)
        time_bin = int(np.floor(onset_rel / 60)) + 1
        
        # REMOVED: Skip calls beyond fixed 6 bins limit
        # if time_bin > 6:
        #     continue
        # NOW: Bins are calculated dynamically based on actual experiment duration from metadata
        
        valid_call_count += 1
        
        # Create call ID
        call_id = f"{chick_id}_call_{valid_call_count:03d}"

        # Store call data (added Experiment_Duration_Sec for tracking)
        call_record = {
            'ChickID': chick_id,
            'Call_ID': call_id,
            'Sex': sex,
            'Biological_Condition': biological_condition,
            'Start_experiment_sec': recorded_start_time,
            'End_experiment_sec': recorded_end_time,
            'Experiment_Duration_Sec': experiment_duration,
            'Call_Number': valid_call_count,
            'Onset_Absolute_Sec': onset,
            'Offset_Absolute_Sec': offset,
            'Onset_Relative_Sec': onset_rel,
            'Offset_Relative_Sec': offset_rel,
            'Duration_call': offset - onset,
            'Time_Bins': time_bin
        }
        all_calls_data.append(call_record)
    
    print(f"Processed {chick_id}: {len(calls_produced_onsets)} total calls, {valid_call_count} calls within experiment window ({experiment_duration:.1f}s)")

# Create final dataframe
if all_calls_data:
    all_calls_df = pd.DataFrame(all_calls_data)
    
    # Check for calls beyond their specific experiment duration
    # NOW USING Experiment_Duration_Sec column from the data itself
    over_limit = all_calls_df[all_calls_df['Offset_Relative_Sec'] > all_calls_df['Experiment_Duration_Sec']]

    if not over_limit.empty:
        print("\nWARNING: Calls that end after experiment duration:")
        print(over_limit[['ChickID', 'Call_ID', 'Onset_Relative_Sec', 'Offset_Relative_Sec', 'Duration_call', 'Experiment_Duration_Sec']].head())
        print(f"Total calls beyond limit: {len(over_limit)}")
    else:
        print("\nAll calls end within the experiment window.")
    
    # Export main CSV
    all_calls_df.to_csv(os.path.join(save_results_path, "all_chicks_calls.csv"), index=False)
    
    print(f"\nAnalysis completed!")
    print(f"All calls saved to: all_chicks_calls.csv")
    print(f"Total calls analyzed: {len(all_calls_df)}")
    print(f"Total chicks: {all_calls_df['ChickID'].nunique()}")
    print(f"Time bins covered: {all_calls_df['Time_Bins'].min()} to {all_calls_df['Time_Bins'].max()}")
    print(f"Experiment durations range: {all_calls_df['Experiment_Duration_Sec'].min():.1f}s to {all_calls_df['Experiment_Duration_Sec'].max():.1f}s")


# ========== EXPORT INDIVIDUAL CHICK FILES (onset-offset format) ==========
print("\n" + "="*70)
print("EXPORTING INDIVIDUAL CHICK FILES")
print("="*70)

# Create folder for individual chick files
individual_files_folder = os.path.join(save_results_path, 'individual_chick_calls')
os.makedirs(individual_files_folder, exist_ok=True)

# Group by ChickID and export
for chick_id in all_calls_df['ChickID'].unique():
    chick_data = all_calls_df[all_calls_df['ChickID'] == chick_id]
    
    # Create DataFrame with only onset and offset columns
    export_data = pd.DataFrame({
        'Onset_Absolute_Sec': chick_data['Onset_Absolute_Sec'],
        'Offset_Absolute_Sec': chick_data['Offset_Absolute_Sec']
    })
    
    # Export as tab-separated text file (like your example)
    output_filename = f"{chick_id}.txt"
    output_path = os.path.join(individual_files_folder, output_filename)
    
    export_data.to_csv(output_path, sep='\t', index=False, header=False, float_format='%.4f')
    
    print(f"Exported: {output_filename} ({len(chick_data)} calls)")

print(f"\nIndividual files saved to: {individual_files_folder}")


# ========== VISUALIZATIONS ==========

# Set publication-quality parameters
plt.style.use('default')  # Start with clean slate
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3
    #bold titles can be set per plot
     

})

# Define consistent color palette for biological conditions
n_conditions = all_calls_df['Biological_Condition'].nunique()
colors = sns.color_palette("Set2", n_conditions)
condition_colors = dict(zip(all_calls_df['Biological_Condition'].unique(), colors))



# Prepare data with corrected time bins (0-based for x-axis)
all_calls_df['Time_Bins_Minutes'] = (all_calls_df['Time_Bins'] - 1).astype(int) # 0-based for plotting

# Determine maximum time bin across all recordings for consistent x-axis
# NOW CALCULATED DYNAMICALLY from actual data instead of hardcoded to 6
max_time_bin = int(all_calls_df['Time_Bins'].max())

print(f"\nVisualization info:")
print(f"Maximum time bin detected: {max_time_bin} (corresponds to {max_time_bin} minutes)")

# ========== FIGURE 1: Total vocalizations per time bin by condition ==========
fig, ax = plt.subplots(figsize=(10, 6))
bin_summary = all_calls_df.groupby(['Time_Bins_Minutes', 'Biological_Condition']).size().reset_index(name='Call_Count')
sns.barplot(data=bin_summary, x='Time_Bins_Minutes', y='Call_Count', 
           hue='Biological_Condition', palette=condition_colors, ax=ax)
ax.set_title('Total Vocalizations per Time Bin', fontweight='bold')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Total Number of Calls')
ax.legend(title='Biological Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

# Customize x-axis labels

ax.set_xticks(range(0, max_time_bin))
ax.set_xticklabels([f'{i}-{i+1}' for i in range(max_time_bin)])

plt.tight_layout()
plt.savefig(os.path.join(save_results_path, 'Figure1_total_vocalizations_per_timebin.png'), 
           dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# ========== FIGURE 2: Individual chick trajectories by condition ==========
fig, ax = plt.subplots(figsize=(12, 7))
chick_bin_summary = all_calls_df.groupby(['ChickID', 'Time_Bins_Minutes', 'Biological_Condition']).size().reset_index(name='Call_Count')

# Calculate mean and SEM for each condition and time bin
avg_bin_summary = chick_bin_summary.groupby(['Time_Bins_Minutes', 'Biological_Condition'])['Call_Count'].agg(['mean', 'sem']).reset_index()
avg_bin_summary.columns = ['Time_Bins_Minutes', 'Biological_Condition', 'Mean_Calls', 'SEM_Calls']

# Plot individual chick trajectories - lighter lines
for condition in chick_bin_summary['Biological_Condition'].unique():
    condition_data = chick_bin_summary[chick_bin_summary['Biological_Condition'] == condition]
    
    for chick_id in condition_data['ChickID'].unique():
        chick_data = condition_data[condition_data['ChickID'] == chick_id]
        ax.plot(chick_data['Time_Bins_Minutes'], chick_data['Call_Count'], 
               color=condition_colors[condition], alpha=0.3, linewidth=1.0)

# Overlay group mean trajectories with SEM error bars
for condition in avg_bin_summary['Biological_Condition'].unique():
    condition_avg = avg_bin_summary[avg_bin_summary['Biological_Condition'] == condition]
    
    # Plot mean line with markers
    ax.plot(condition_avg['Time_Bins_Minutes'], condition_avg['Mean_Calls'], 
           color=condition_colors[condition], linewidth=4, marker='o', markersize=10,
           markeredgecolor='white', markeredgewidth=2, label=f'{condition} (Mean ± SEM)',
           alpha=1.0)
    
    # Add SEM error bars
    ax.errorbar(condition_avg['Time_Bins_Minutes'], condition_avg['Mean_Calls'], 
               yerr=condition_avg['SEM_Calls'], color=condition_colors[condition],
               capsize=5, capthick=2, linewidth=2, alpha=0.8, fmt='none')

ax.set_title('Individual Chick Vocal Trajectories with Group Means ± SEM', fontweight='bold')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Calls per Individual')
ax.legend(title='Biological Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

# Customize x-axis labels (DYNAMIC VERSION)
ax.set_xticks(range(0, max_time_bin))
ax.set_xticklabels([f'{i}-{i+1}' for i in range(max_time_bin)])

plt.tight_layout()
plt.savefig(os.path.join(save_results_path, 'Figure2_mean_vocalizations_per_individual.png'), 
           dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# ========== FIGURE 3: Individual trajectories by condition ==========
fig, ax = plt.subplots(figsize=(12, 7))

# Use the original data structure for this plot
chick_bin_summary = all_calls_df.groupby(['ChickID', 'Time_Bins_Minutes', 'Biological_Condition']).size().reset_index(name='Call_Count')
simple_avg_summary = chick_bin_summary.groupby(['Time_Bins_Minutes', 'Biological_Condition'])['Call_Count'].mean().reset_index()

# Plot individual trajectories with transparency
sns.lineplot(data=chick_bin_summary, x='Time_Bins_Minutes', y='Call_Count', 
            hue='Biological_Condition', alpha=0.3, ax=ax, palette=condition_colors, legend=False)

# Overlay mean trajectories with thicker lines
sns.lineplot(data=simple_avg_summary, x='Time_Bins_Minutes', y='Call_Count', 
            hue='Biological_Condition', marker='o', linewidth=3, ax=ax, palette=condition_colors, markersize=8)

ax.set_title('Individual Vocal Trajectories with Group Means', fontweight='bold')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Calls per Individual')
ax.legend(title='Biological Condition', bbox_to_anchor=(1.05, 1), loc='upper left')


ax.set_xticks(range(0, max_time_bin))
ax.set_xticklabels([f'{i}-{i+1}' for i in range(max_time_bin)])

plt.tight_layout()
plt.savefig(os.path.join(save_results_path, 'Figure3_individual_trajectories.png'), 
           dpi=600, bbox_inches='tight', facecolor='white')



plt.show()


# ========== FIGURE 3: Individual trajectories by condition ==========
fig, ax = plt.subplots(figsize=(12, 7))

# Use the original data structure for this plot
chick_bin_summary = all_calls_df.groupby(['ChickID', 'Time_Bins_Minutes', 'Biological_Condition']).size().reset_index(name='Call_Count')
simple_avg_summary = chick_bin_summary.groupby(['Time_Bins_Minutes', 'Biological_Condition'])['Call_Count'].mean().reset_index()

# Plot individual trajectories with transparency
sns.lineplot(data=chick_bin_summary, x='Time_Bins_Minutes', y='Call_Count', 
            hue='Biological_Condition', alpha=0.3, ax=ax, palette=condition_colors, legend=False)

# Overlay mean trajectories with thicker lines
sns.lineplot(data=simple_avg_summary, x='Time_Bins_Minutes', y='Call_Count', 
            hue='Biological_Condition', marker='o', linewidth=3, ax=ax, palette=condition_colors, markersize=8)

ax.set_title('Individual Vocal Trajectories with Group Means', fontweight='bold', pad=40)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Calls per Individual')

# Position legend at the top, below the title
ax.legend(title='Biological Condition', 
         loc='upper center', 
         bbox_to_anchor=(0.5, 1.12),
         ncol=len(condition_colors),  # Display legend items horizontally
         frameon=True,
         fancybox=False,
         shadow=False)

ax.set_xticks(range(0, max_time_bin))
ax.set_xticklabels([f'{i}-{i+1}' for i in range(max_time_bin)])

plt.tight_layout()
plt.savefig(os.path.join(save_results_path, 'Figure3_individual_trajectories.png'), 
           dpi=600, bbox_inches='tight', facecolor='white')
plt.show()


# ========== FIGURE 4: Box plot - Total vocalizations per session ==========
session_summary = all_calls_df.groupby(['ChickID', 'Biological_Condition', 'Sex']).size().reset_index(name='Total_Calls')

fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(data=session_summary, x='Biological_Condition', y='Total_Calls', 
           palette=condition_colors, ax=ax)
sns.stripplot(data=session_summary, x='Biological_Condition', y='Total_Calls', 
             color='black', alpha=0.7, ax=ax, size=6)
ax.set_title('Total Vocalizations per Session by Condition', fontweight='bold')
ax.set_xlabel('Biological Condition')
ax.set_ylabel('Total Calls per Individual')

plt.tight_layout()
plt.savefig(os.path.join(save_results_path, 'Figure4_total_calls_boxplot.png'), 
           dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# ========== FIGURE 5: Violin plot - Distribution of vocal activity ==========
fig, ax = plt.subplots(figsize=(10, 7))
sns.violinplot(data=session_summary, x='Biological_Condition', y='Total_Calls',
              palette=condition_colors, ax=ax)
ax.set_title('Distribution of Vocal Activity by Condition', fontweight='bold')
ax.set_xlabel('Biological Condition')
ax.set_ylabel('Total Calls per Individual')

plt.tight_layout()
plt.savefig(os.path.join(save_results_path, 'Figure5_distribution_violinplot.png'), 
           dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# ========== FIGURE 6: Sex differences (if available) ==========
fig, ax = plt.subplots(figsize=(10, 7))
if session_summary['Sex'].nunique() > 1 and 'Unknown' not in session_summary['Sex'].unique():
    sns.barplot(data=session_summary, x='Biological_Condition', y='Total_Calls', hue='Sex', ax=ax)
    ax.set_title('Sex Differences in Vocal Activity', fontweight='bold')
    ax.set_xlabel('Biological Condition')
    ax.set_ylabel('Mean Calls per Individual')
    ax.legend(title='Sex', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_results_path, 'Figure6_sex_differences.png'), 
               dpi=600, bbox_inches='tight', facecolor='white')
else:
    ax.text(0.5, 0.5, 'Sex data not available\nor insufficient variation', 
           ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Sex Analysis', fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_results_path, 'Figure6_sex_analysis_unavailable.png'), 
               dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# ========== SUMMARY STATISTICS TABLE ==========
print("\n" + "="*70)
print("PUBLICATION-READY SUMMARY STATISTICS")
print("="*70)

# Create comprehensive summary table
summary_stats = []

# Overall statistics
overall_stats = {
    'Group': 'Overall',
    'N_Individuals': all_calls_df['ChickID'].nunique(),
    'N_Calls': len(all_calls_df),
    'Mean_Calls_Per_Individual': len(all_calls_df) / all_calls_df['ChickID'].nunique(),
    'SD_Calls_Per_Individual': session_summary['Total_Calls'].std(),
    'Mean_Call_Duration': all_calls_df['Duration_call'].mean(),
    'SD_Call_Duration': all_calls_df['Duration_call'].std()
}
summary_stats.append(overall_stats)

# By biological condition
for condition in session_summary['Biological_Condition'].unique():
    condition_sessions = session_summary[session_summary['Biological_Condition'] == condition]
    condition_calls = all_calls_df[all_calls_df['Biological_Condition'] == condition]
    
    condition_stats = {
        'Group': f'{condition}',
        'N_Individuals': len(condition_sessions),
        'N_Calls': len(condition_calls),
        'Mean_Calls_Per_Individual': condition_sessions['Total_Calls'].mean(),
        'SD_Calls_Per_Individual': condition_sessions['Total_Calls'].std(),
        'Mean_Call_Duration': condition_calls['Duration_call'].mean(),
        'SD_Call_Duration': condition_calls['Duration_call'].std()
    }
    summary_stats.append(condition_stats)

# Convert to DataFrame and format for publication
summary_df = pd.DataFrame(summary_stats)
summary_df = summary_df.round(2)

print("\nTable 1: Descriptive Statistics of Vocal Behavior")
print("-" * 70)
print(f"{'Group':<15} {'N_Ind':<6} {'N_Calls':<8} {'Calls/Ind':<10} {'±SD':<8} {'Duration':<10} {'±SD':<8}")
print("-" * 70)

for _, row in summary_df.iterrows():
    print(f"{row['Group']:<15} {row['N_Individuals']:<6.0f} {row['N_Calls']:<8.0f} "
          f"{row['Mean_Calls_Per_Individual']:<10.2f} {row['SD_Calls_Per_Individual']:<8.2f} "
          f"{row['Mean_Call_Duration']:<10.3f} {row['SD_Call_Duration']:<8.3f}")

# Export summary table
summary_df.to_csv(os.path.join(save_results_path, 'Table1_summary_statistics.csv'), index=False)

print(f"\nFigures and summary table saved to: {save_results_path}")