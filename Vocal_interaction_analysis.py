import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import onsets_offsets_detection_utils as onset_detectors
from onsets_offsets_detection_utils import offset_detection_first_order
from utils_interaction_analysis import load_stimulus_log_absolute, group_stimuli_sequences_simple, filter_calls_produced
from utils_interaction_analysis import map_stimulus_to_condition, create_comprehensive_dataset, calculate_intercalls_intervals_stat
from utils_interaction_analysis import calculate_sequence_latency_response, latency_results_export

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Load file paths
# audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\Normalised_Audio'
# stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Stimuli_log_vocal_echo\Testing_1'
# metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2025.05.27\metadata_testing_1.csv'

# audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2'
# stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2\Stimuli_log'
# metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_1_2\metadata_testing.csv'


# audio_folder = r'C:\Users\anton\OneDrive\Documenti\Automatic_testing\Normalised_audio'
# stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Automatic_testing\Stimuli_log'
# metadata_path = r'C:\Users\anton\OneDrive\Documenti\Automatic_testing\metadata_testing_automatic.csv'


audio_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_automatic\Testing_automatic'
stimuli_folder = r'C:\Users\anton\OneDrive\Documenti\Testing_automatic\Stimuli_log'
metadata_path = r'C:\Users\anton\OneDrive\Documenti\Testing_automatic\metadata_automatic_testing.csv'


# Output directory setup
save_results_path = os.path.join(audio_folder, 'analysis_results')
os.makedirs(save_results_path, exist_ok=True)

# Load metadata and standardise column names
metadata = pd.read_csv(metadata_path)

# Clean column headers (remove extra spaces and standardise names)
metadata.columns = metadata.columns.str.strip()
if 'Trial sequence' in metadata.columns:
    metadata = metadata.rename(columns={'Trial sequence': 'Trial_Sequence'})

# Clean string columns (remove extra spaces)
string_columns = ['Chick_ID', 'Filename', 'Condition', 'Trial_Sequence', 'Stimulus_log_filename']
for col in string_columns:
    if col in metadata.columns:
        metadata[col] = metadata[col].astype(str).str.strip()

print(f"Loaded metadata for {len(metadata)} recordings")



# HFC parameters 
# HFC_parameters = {
#     'hop_length': 441, 'sr': 44100, 'spec_num_bands': 32, 'spec_fmin': 1000, 'spec_fmax': 15000, 'spec_fref': 3200,
#     'pp_threshold': 1.4, 'pp_pre_avg': 25, 'pp_post_avg': 40, 'pp_pre_max': 1, 'pp_post_max': 1,
#     'global_shift': 0.10, 'double_onset_correction': 0.1
# }



HFC_parameters = {
    'hop_length': 441, 'sr': 44100, 'spec_num_bands': 24, 'spec_fmin': 1500, 'spec_fmax': 10000, 'spec_fref': 3000,
    'pp_threshold': 0.5, 'pp_pre_avg': 25, 'pp_post_avg': 1, 'pp_pre_max': 1, 'pp_post_max': 1,
    'global_shift': 0.070, 'double_onset_correction': 0.1
}


# Main loop
all_results = []
all_results_latency = pd.DataFrame()



audio_files = glob.glob(os.path.join(audio_folder, '*.wav'))

print(f"Total of {len(audio_files)} audio files to process")

for audio_file in tqdm(audio_files, desc="Processing audio files"):
    base_name = os.path.basename(audio_file)

    # Metadata already contains filenames with .wav extension
    meta_row = metadata[metadata['Filename'] == base_name]
    
    if meta_row.empty:
        print(f"[SKIP] No metadata found for file: {base_name}")
        continue

    # Extract metadata for this recording
    meta_row = meta_row.iloc[0]
    
    # Extract chick ID from filename (format: Chick_X_YYYY.MM.DD)
    chick_id = os.path.splitext(base_name)[0]
    experiment_date = str(meta_row['Experiment_date']).replace("/", "-")
    unique_id = f"{chick_id}_{experiment_date}"  # Create unique identifier combining ID and date
    recorded_start_time = float(meta_row['Start_experiment_sec'])
    stim_log_filename = meta_row['Stimulus_log_filename']
    stim_log_path = os.path.join(stimuli_folder, stim_log_filename)
    biological_condition = meta_row['Condition']
    sequence_type = meta_row['Trial_Sequence']

    recorded_end_time = float(meta_row['End_experiment_sec'])


    # Check if stimulus log file is there
    if not os.path.exists(stim_log_path):
        print(f"Stimulus log not found: {stim_log_path}")
        continue

    stimuli = load_stimulus_log_absolute(stim_log_path, recorded_start_time)

    valid_stimuli = []
    for s in stimuli:
        t = s['time']  # tempo assoluto in secondi

        # opzionale: durata stimolo, se ti serve includerla
        stim_dur = 0.2  # es: 200 ms, oppure guarda dal dizionario per quella condizione

        pre_start  = t - 10
        post_end   = t + stim_dur + 10

        if (pre_start >= recorded_start_time) and (post_end <= recorded_end_time):
            valid_stimuli.append(s)

    stimuli = valid_stimuli

    sequences = group_stimuli_sequences_simple(stimuli, biological_condition)

    # Detect calls
    # Extract HFC parameters (exclude correction parameters for initial detection)
    hfc_params = {k: v for k, v in HFC_parameters.items() if k not in ['global_shift', 'double_onset_correction']}
    # Detect call onsets using HFC method
    hfc_onsets = onset_detectors.high_frequency_content(audio_file, visualise_activation=False, **hfc_params)
    hfc_onsets = onset_detectors.global_shift_correction(hfc_onsets, HFC_parameters['global_shift'])
    hfc_onsets = onset_detectors.double_onset_correction(hfc_onsets, correction=HFC_parameters['double_onset_correction'])
    
    # Detect call offsets based on onsets
    predicted_offsets = offset_detection_first_order(audio_file, hfc_onsets)
    # calls_produced_onsets, calls_produced_offsets = filter_calls_produced(
    #     hfc_onsets, predicted_offsets, stimuli, buffer=0.1
    # )

    # Updated call to filter_calls_produced with biological_condition parameter (no buffer)
    calls_produced_onsets, calls_produced_offsets = filter_calls_produced(
        hfc_onsets, predicted_offsets, stimuli, biological_condition
    )

    #  Filter calls within experimental window (10 minutes or different; the code will take the start and end from metadata)
    experiment_duration = recorded_end_time - recorded_start_time
    valid_mask = (calls_produced_onsets >= recorded_start_time) & \
                 (calls_produced_offsets <= recorded_end_time)
    
    # convert to numpy arrays for indexing
    calls_produced_onsets = np.array(calls_produced_onsets)
    calls_produced_offsets = np.array(calls_produced_offsets)

    calls_produced_onsets = calls_produced_onsets[valid_mask]
    calls_produced_offsets = calls_produced_offsets[valid_mask]
    
    print(f"  {chick_id}: {len(hfc_onsets)} total detections → "
          f"{len(calls_produced_onsets)} calls within experiment window ({experiment_duration:.1f}s)")


# Calculate latency to response for each sequence
    latency_results = calculate_sequence_latency_response(calls_produced_onsets, sequences)

    # Enrich and prepare latency data for export
    latency_results = latency_results_export(
        latency_results=latency_results,
        biological_condition=biological_condition,
        chick_id=chick_id,
        experiment_date=experiment_date,
        sequence_type=sequence_type
    )

    # Append to global table
    latency_results_df = pd.DataFrame(latency_results)
    all_results_latency = pd.concat([all_results_latency, latency_results_df], ignore_index=True)


    # Add to dataset 
    chick_data = create_comprehensive_dataset(
        calls_produced_onsets, calls_produced_offsets, stimuli, sequences, chick_id, biological_condition, sequence_type
    )
    # Add additional metadata to each record
    for record in chick_data:
        record['Experiment_Date'] = experiment_date

    # Calculate Inter-Call Intervals (ICI) for the entire session
    advanced_metrics = calculate_intercalls_intervals_stat(
    calls_produced_onsets, calls_produced_offsets, sequences
    )

    
    # Add advanced metrics to all records for this chick
    for record in chick_data:
        record.update(advanced_metrics)

    all_results.extend(chick_data)
    
    print(f"Processed {chick_id}: {len(calls_produced_onsets)} calls, "
          f"{len(sequences)} sequences, sequence type: {sequence_type}")
    print(f"  Advanced metrics: {advanced_metrics}")


# After processing all files, save combined latency data across all chicks
if not all_results_latency.empty:
    latency_output_path = os.path.join(save_results_path, 'all_chicks_latency_responses.csv')
    all_results_latency.to_csv(latency_output_path, index=False)
    print(f"\nSaved combined latency responses → {latency_output_path}")
    print(f"Total latency records: {len(all_results_latency)}")
else:
    print("\nNo latency responses found to save.")



# Convert to DataFrame and prepare for statistical analysis
final_dataset = pd.DataFrame(all_results)

if not final_dataset.empty:
    # Create additional variables for statistical modelling
    final_dataset['Trial_Order_Centered'] = final_dataset['Trial_Order'] - final_dataset['Trial_Order'].mean()
    final_dataset['Stimulus_Type_Factor'] = pd.Categorical(final_dataset['Stimulus_Type'])
    final_dataset['Biological_Condition_Factor'] = pd.Categorical(final_dataset['Biological_Condition'])
    final_dataset['Sequence_Type_Factor'] = pd.Categorical(final_dataset['Sequence_Type'])

    
    
    # Log-transform Call Suppression Ratio (add small constant to avoid log(0))
    final_dataset['Log_CSR'] = np.log(final_dataset['Call_Suppression_Ratio'] + 0.001)

    # Export main dataset
    output_file = os.path.join(save_results_path, 'comprehensive_dataset_for_R.csv')
    final_dataset.to_csv(output_file, index=False)

    # Generate and export summary statistics
    summary_stats = final_dataset.groupby(['Biological_Condition', 'Stimulus_Type']).agg({
        'Call_Suppression_Ratio': ['count', 'mean', 'std', 'median'],
        'Calls_Pre_10s': ['mean', 'std'],
        'Calls_Post_10s': ['mean', 'std'],
        'Total_Calls_Session': ['mean', 'std']
    }).round(3)
    summary_stats.to_csv(os.path.join(save_results_path, 'summary_statistics.csv'))

        # Generate and export ICI statistics by biological condition
    ici_stats = final_dataset.groupby('Biological_Condition')['Mean_ICI'].agg([
        'count', 'mean', 'std', 'median'
    ]).round(3)
    ici_stats.to_csv(os.path.join(save_results_path, 'ici_by_biological_condition.csv'))


    print(f"Final dataset exported to: {output_file}")
    print(f"Dataset shape: {final_dataset.shape}")
    print(f"Columns: {list(final_dataset.columns)}")



    print(f"\nSample counts by condition and stimulus type:")
    condition_summary = final_dataset.groupby(['Biological_Condition', 'Stimulus_Type']).size()
    print(condition_summary)

    print(f"\nInter-Call Interval (ICI) statistics by biological condition:")
    print(ici_stats)

else:
    print(f"No valid data found for analysis.")

# visualisations
# =========================
plt.style.use('default')
mpl.rcParams.update({
    "font.family": "Times New Roman",
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
})
plt.rcParams["font.family"] = "Times New Roman"

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

# Output folder for figures
viz_folder = os.path.join(save_results_path, "figures_single")
os.makedirs(viz_folder, exist_ok=True)

# =========================
# FIGURE 1 — Total calls per session by biological condition (box plot)
# =========================
fig = plt.figure(figsize=(8, 5.5))
ax = plt.gca()
sns.boxplot(
    data=final_dataset,
    x='Biological_Condition', y='Total_Calls_Session',
    palette='Set2', ax=ax, showcaps=True, whis=1.5, fliersize=0
)
ax.set_title('Total calls per session by biological condition')
ax.set_xlabel('Biological condition')
ax.set_ylabel('Total calls (per session)')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure1_total_calls_by_condition.png'))
# plt.savefig(os.path.join(viz_folder, 'Figure1_total_calls_by_condition.pdf'))
# plt.show()

# =========================
# FIGURE 2 — Call Suppression Ratio by biological condition (violin)
# =========================
fig = plt.figure(figsize=(8, 5.5))
ax = plt.gca()
sns.violinplot(
    data=final_dataset,
    x='Biological_Condition', y='Call_Suppression_Ratio',
    palette='viridis', inner='quartile', cut=0, ax=ax
)
ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (CSR = 1)')
ax.legend(frameon=True, title=None)
ax.set_title('Call Suppression Ratio by biological condition')
ax.set_xlabel('Biological condition')
ax.set_ylabel('CSR')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure2_csr_by_condition.png'))
# plt.show()



# =========================
# FIGURE 3 — Inter-Call Interval distribution (histogram with KDE)
# =========================
fig = plt.figure(figsize=(8, 5.5))
ax = plt.gca()
sns.histplot(
    data=final_dataset, x='Mean_ICI', hue='Biological_Condition',
    kde=True, alpha=0.3, ax=ax
)
ax.set_title('Inter-Call Interval distribution')
ax.set_xlabel('Mean ICI (s)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure3_ici_distribution.png'))
# plt.show()


# pre post Inter-Call Interval (ICI) Pre vs Post
# -----------------------------
# PRE–POST ICI: data prep
# -----------------------------
ici_long = (
    final_dataset[['Biological_Condition', 'Mean_ICI_Pre_Stim', 'Mean_ICI_Post_Stim']]
    .copy()
    .rename(columns={
        'Mean_ICI_Pre_Stim': 'ICI_Pre',
        'Mean_ICI_Post_Stim': 'ICI_Post'
    })
)

# Long format for bar plot
ici_long = ici_long.melt(
    id_vars='Biological_Condition',
    value_vars=['ICI_Pre', 'ICI_Post'],
    var_name='Phase', value_name='Mean_ICI'
)
ici_long['Phase'] = ici_long['Phase'].map({'ICI_Pre': 'Pre', 'ICI_Post': 'Post'})
ici_long = ici_long.dropna(subset=['Mean_ICI'])

# (optional) ensure a consistent order of conditions on the x-axis
cond_order = sorted(ici_long['Biological_Condition'].dropna().unique())

# Create output folder (single-figure set)
viz_folder = os.path.join(save_results_path, "figures_single")
os.makedirs(viz_folder, exist_ok=True)

# -----------------------------
# FIGURE A — Pre vs Post ICI (bar, mean ± 95% CI)
# -----------------------------
fig = plt.figure(figsize=(8.5, 5.5))
ax = plt.gca()
sns.barplot(
    data=ici_long, x='Biological_Condition', y='Mean_ICI', hue='Phase',
    order=cond_order, estimator=np.mean, errorbar='ci', ci=95,
    palette=["#90CAF9", "#0D47A1"], ax=ax
)
ax.set_title('Pre vs Post Inter-Call Interval (mean ± 95% CI)', fontname='Times New Roman')
ax.set_xlabel('Biological condition', fontname='Times New Roman')
ax.set_ylabel('Mean ICI (s)', fontname='Times New Roman')
ax.legend(title='Phase', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure_ici_pre_vs_post.png'))
# plt.show()

# -----------------------------
# FIGURE B — Percentage change of ICI by condition (violin)
# (Post–Pre)/Pre * 100; row-wise, then visualised per condition
# -----------------------------
ici_pc = final_dataset[['Biological_Condition', 'Mean_ICI_Pre_Stim', 'Mean_ICI_Post_Stim']].copy()
ici_pc = ici_pc.dropna(subset=['Mean_ICI_Pre_Stim', 'Mean_ICI_Post_Stim'])

# Guard against division by zero (add a tiny epsilon)
eps = 1e-9
ici_pc['ICI_Percent_Change'] = (
    (ici_pc['Mean_ICI_Post_Stim'] - ici_pc['Mean_ICI_Pre_Stim']) /
    (ici_pc['Mean_ICI_Pre_Stim'] + eps)
) * 100.0

fig = plt.figure(figsize=(8.0, 5.5))
ax = plt.gca()
sns.violinplot(
    data=ici_pc, x='Biological_Condition', y='ICI_Percent_Change',
    order=cond_order, palette='viridis', inner='quartile', cut=0, ax=ax
)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax.set_title('Percentage change of Inter-Call Interval (Post vs Pre)', fontname='Times New Roman')
ax.set_xlabel('Biological condition', fontname='Times New Roman')
ax.set_ylabel('Percentage change of ICI (%)', fontname='Times New Roman')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure_ici_percent_change.png'))
# plt.show()



# =========================
# FIGURE 4 — Pre vs Post stimulus calls (bar plot with phase as hue)
# =========================
pre_post_records = []
for _, r in final_dataset.iterrows():
    pre_post_records.append({'Condition': r['Biological_Condition'], 'Phase': 'Pre',  'Calls': r['Calls_Pre_10s']})
    pre_post_records.append({'Condition': r['Biological_Condition'], 'Phase': 'Post', 'Calls': r['Calls_Post_10s']})
pre_post_df = pd.DataFrame(pre_post_records)

fig = plt.figure(figsize=(8.5, 5.5))
ax = plt.gca()
sns.barplot(
    data=pre_post_df, x='Condition', y='Calls', hue='Phase',
    palette='pastel', errorbar='ci', ci=95, estimator=np.mean, ax=ax
)
ax.set_title('Pre vs Post stimulus (mean ± 95% CI)')
ax.set_xlabel('Biological condition')
ax.set_ylabel('Calls (10 s window)')
ax.legend(title='Phase', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure4_pre_vs_post_calls.png'))
# plt.show()

# =========================
# FIGURE 5 — CSR by sequence type (box plot, hue = condition)
# =========================
fig = plt.figure(figsize=(9, 5.5))
ax = plt.gca()
sns.boxplot(
    data=final_dataset,
    x='Sequence_Type', y='Call_Suppression_Ratio',
    hue='Biological_Condition', showcaps=True, whis=1.5, fliersize=0, ax=ax
)
ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax.set_title('CSR by sequence type')
ax.set_xlabel('Sequence type')
ax.set_ylabel('CSR')
ax.legend(title='Biological condition', frameon=True, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure5_csr_by_sequence_type.png'))
# plt.show()

# =========================
# FIGURE 6 — Percentage change after stimulus (violin)
# =========================
final_dataset = final_dataset.copy()
final_dataset['Diff_Post_Pre'] = final_dataset['Calls_Post_10s'] - final_dataset['Calls_Pre_10s']
final_dataset['Perc_Change'] = ((final_dataset['Calls_Post_10s'] - final_dataset['Calls_Pre_10s']) /
                                (final_dataset['Calls_Pre_10s'] + 1)) * 100  # avoid division by zero

fig = plt.figure(figsize=(8, 5.5))
ax = plt.gca()
sns.violinplot(
    data=final_dataset, x='Biological_Condition', y='Perc_Change',
    palette='viridis', inner='quartile', cut=0, ax=ax
)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax.set_title('Percentage change after stimulus')
ax.set_xlabel('Biological condition')
ax.set_ylabel('Percentage change (%)')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure6_percentage_change_post_stimulus.png'))
# plt.show()

# =========================
# FIGURE 7 — CSR distribution with reference thresholds (histogram + KDE)
# =========================
fig = plt.figure(figsize=(8, 5.5))
ax = plt.gca()
sns.histplot(
    data=final_dataset, x='Call_Suppression_Ratio', hue='Biological_Condition',
    kde=True, alpha=0.6, ax=ax
)
ax.axvline(x=1,   color='red',    linestyle='--', alpha=0.8, label='Baseline (CSR = 1)')
ax.axvline(x=0.5, color='orange', linestyle=':',  alpha=0.8, label='50% suppression')
ax.axvline(x=2,   color='green',  linestyle=':',  alpha=0.8, label='100% increase')
ax.set_title('CSR distribution with reference thresholds')
ax.set_xlabel('CSR')
ax.set_ylabel('Count')
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, 'Figure7_csr_distribution_with_thresholds.png'))
# plt.show()

# =========================
# FIGURE 8+ — Individual patterns (one figure per chick)
# =========================
# Create one figure per chick (up to 12 to keep output manageable)
# Select up to 18 chicks to display
sample_chicks = final_dataset['Chick_ID'].dropna().unique()[:18]
sample_data = final_dataset[final_dataset['Chick_ID'].isin(sample_chicks)].copy()

# Create grid of subplots (3 rows × 6 columns = 18 panels)
n_rows, n_cols = 3, 6
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12))
fig.suptitle('Individual CSR patterns across trials', fontsize=16, y=0.95)

for i, chick in enumerate(sample_chicks):
    r, c = divmod(i, n_cols)
    ax = axes[r, c]
    subset = sample_data[sample_data['Chick_ID'] == chick].sort_values("Trial_Order")
    
    sns.lineplot(
        data=subset, x='Trial_Order', y='Call_Suppression_Ratio',
        hue='Biological_Condition', marker='o', ax=ax, legend=False
    )
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.6)
    ax.set_title(f'Chick {chick}', fontsize=10)
    ax.set_xlabel('Trial order')
    ax.set_ylabel('CSR')
    ax.set_ylim(0, max(3, subset['Call_Suppression_Ratio'].max() * 1.1))

# Remove unused axes if less than 12 chicks
for j in range(len(sample_chicks), n_rows * n_cols):
    r, c = divmod(j, n_cols)
    axes[r, c].axis("off")

# Add one global legend (take handles from first subplot with data)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Biological condition",
           loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig(os.path.join(viz_folder, 'Figure8_individual_patterns_grid.png'))
# plt.show()


print(f"\nFigures saved to: {viz_folder}")

