import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Function to load audio files from a directory
# Returns a dictionary with recording names as keys and (audio data, sample rate) as values
def load_audio_data(path):
    audios = {}
    for f in os.listdir(path):
        if f.lower().endswith('.wav'):
            key = os.path.splitext(f)[0]
            y, sr = librosa.load(os.path.join(path, f), sr=None)
            audios[key] = (y, sr)
    return audios

def plot_single_call(row, audio_data, out_path, title=None):
    rec = row['recording']
    onset, offset = row['onsets_sec'], row['offsets_sec']
    y_full, sr = audio_data.get(rec, (None, None))
    if y_full is None:
        raise FileNotFoundError(f"{rec}.wav non trovato.")

    start, end = int(onset * sr), int(offset * sr)
    y = y_full[start:end]
    t = np.linspace(0, (end-start)/sr, len(y))

    # Prepare feature text
    metadata_cols = [
        'recording', 'onsets_sec', 'offsets_sec', 'distance_to_center',
        'cluster_membership', 'label', 'call_id', 'Call Number'
    ]
    feat_cols = [c for c in row.index if c not in metadata_cols]
    feats = row[feat_cols].to_dict()
    feature_text = "\n".join(f"{k}: {v:.2f}" for k, v in feats.items())

    # create the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=600,
                             facecolor='black',
                             gridspec_kw={'width_ratios':[1.2, 2.0, 1.5]})

    # 1) Waveform plot
    ax_w = axes[0]
    ax_w.set_facecolor('white')
    ax_w.plot(t, y, linewidth=0.7, color='blue')
    ax_w.set_title('Waveform', fontsize=10, family='Times New Roman', color='white')
    ax_w.set_xlabel('Time (s)', fontsize=6, family='Times New Roman', color='white')
    ax_w.set_ylabel('Amplitude', fontsize=6, family='Times New Roman', color='white')
    for spine in ax_w.spines.values():
        spine.set_color('white')
    ax_w.tick_params(colors='white', labelsize=6)

    # 2) Mel-spectrogram plot
    ax_s = axes[1]
    ax_s.set_facecolor('black')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                       fmin=2000, fmax=12000, power=1.0)
    logS = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(logS, sr=sr,
                             x_axis='time', y_axis='mel',
                             cmap='magma', ax=ax_s)
    ax_s.set_title(f'{rec} ({onset:.2f}-{offset:.2f}s)',
                   fontsize=10, color='white', family='Times New Roman')
    ax_s.set_xlabel('Time (s)', fontsize=6, family='Times New Roman', color='white')
    ax_s.set_ylabel('Mel freq (Hz)', fontsize=6, family='Times New Roman', color='white')
    for spine in ax_s.spines.values():
        spine.set_color('white')
    ax_s.tick_params(colors='white', labelsize=6)

    # 3) Feature statistics to text along with the calls
    ax_f = axes[2]
    ax_f.set_facecolor('black')
    ax_f.text(0.01, 0.98, feature_text,
              va='top', ha='left', fontsize=9.5,
              family='Times New Roman', color='white')
    ax_f.axis('off')


    if title:
        fig.suptitle(title, y=1.02,
                     fontsize=12, color='white', family='Times New Roman')
    fig.tight_layout(pad=1)

    # Save the figure
    fig.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Salvato {out_path}")




def call_spectrogram_generator(df, audio_data, output_dir,
                               spec_cmap='magma', fmin=2000, fmax=12000, dpi=600):


    os.makedirs(output_dir, exist_ok=True)

    yticks = [2048, 4096, 8192, 12000]

    for idx, row in df.iterrows():
        rec = row['recording']
        onset, offset = row['onsets_sec'] , row['offsets_sec'] # + 00.050
        y_full, sr = audio_data.get(rec, (None, None))
        if y_full is None:
            print(f"{rec}.wav non trovato in audio_data")
            continue

        start, end = int(onset * sr), int(offset * sr)
        y = y_full[start:end]
        duration = (end - start) / sr

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=fmin, fmax=fmax)
        log_S = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(2.6, 2), dpi=dpi, facecolor='white')

        ax.set_facecolor('black')
        img = librosa.display.specshow(
            log_S, sr=sr, x_axis='time', y_axis='mel',
            cmap=spec_cmap, ax=ax, fmin=fmin, fmax=fmax
        )


        ax.set_title(
            f'Call from cluster {row["cluster_membership"]} - {rec}',
            fontsize=10, color='black', family='Times New Roman'
        )
        ax.set_xlabel('Time (s)', fontsize=8, color='black', family='Times New Roman')
        ax.set_ylabel('Mel freq (Hz)', fontsize=8, color='black', family='Times New Roman')
        for spine in ax.spines.values():
            spine.set_color('white')

        # --- Set X ticks to fixed values ---
        xlim = ax.get_xlim()
        xticks = np.linspace(xlim[0], xlim[1], num=5)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelsize=7, colors='black')
        for label in ax.get_xticklabels():
            label.set_fontname('Times New Roman')

        # --- Set Y ticks to fixed values ---
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=7, colors='black')
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        ax.grid(False)

        plt.tight_layout(pad=1.1)
        out_name = f"{rec}_call_{idx+1}.png"
        plt.savefig(os.path.join(output_dir, out_name), bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Salvato {os.path.join(output_dir, out_name)}")


# Batch per cluster
def make_identikit_for_cluster(df, audio_data, out_dir, cluster_id, top_n=5):
    df_top = df.nsmallest(top_n, 'distance_to_center').reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    for idx, row in df_top.iterrows():
        filename = f'cluster{cluster_id}_idx{idx}_call.png'
        title = f'Cluster {cluster_id} - Call {idx}'
        plot_single_call(row, audio_data,
                         os.path.join(out_dir, filename), title)
        
        call_spectrogram_generator(df_top, audio_data, out_dir)

# MAIN
if __name__ == '__main__':
    audio_path = r"C:\User\Documents\audio_path"
    dict_path  = r"C:\User\Documents\Results\clustering_results_path"  # load the csv file with the calls id
    output_folder = r"C:\Users\\Documents\Results\Identikit_Output"

    print("Loading audio data…")
    audio_data = load_audio_data(audio_path)

    cluster_numbers = [0, 1, 2]  # Example cluster IDs


    for cl in cluster_numbers:
        df = pd.read_csv(os.path.join(dict_path, f"dictionary_cluster_{cl}.csv"))
        make_identikit_for_cluster(df, audio_data,
                                   os.path.join(output_folder, f"cluster_{cl}"),
                                   cluster_id=cl, top_n=5)
